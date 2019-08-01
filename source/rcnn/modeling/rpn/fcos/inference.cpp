#include "rpn/fcos/inference.h"

namespace rcnn
{
namespace modeling
{

FCOSPostProcessorImpl::FCOSPostProcessorImpl(float pre_nms_thresh, int pre_nms_top_n,
                                             float nms_thresh, int fpn_post_num_top_n, int min_size,
                                             int num_classes)
    : pre_nms_thresh(pre_nms_thresh), pre_nms_top_n(pre_nms_top_n),
      nms_thresh(nms_thresh), fpn_post_num_top_n(fpn_post_num_top_n),
      min_size(min_size), num_classes(num_classes)

{
}

std::vector<structures::BoxList> FCOSPostProcessorImpl::forward(std::vector<torch::Tensor> &locations,
                                                                std::vector<torch::Tensor> &box_cls,
                                                                std::vector<torch::Tensor> &box_regression,
                                                                std::vector<torch::Tensor> &centerness,
                                                                std::vector<std::pair<int64_t, int64_t>> image_sizes)
{
    using namespace structures;
    std::vector<std::vector<BoxList>> sampled_boxes;
    for (size_t i = 0; i < locations.size(); ++i)
    {
        auto l = locations[i];
        auto o = box_cls[i];
        auto b = box_regression[i];
        auto c = centerness[i];

        sampled_boxes.push_back(this->forward_for_single_feature_map(l, o, b, c, image_sizes));
    }

    std::vector<BoxList> results;
    for (auto &b : sampled_boxes)
    {
        BoxList br = BoxList::CatBoxList(b);
        results.push_back(br);
    }
    auto boxlists = this->select_over_all_levels(results);

    return boxlists;
}

std::vector<structures::BoxList> FCOSPostProcessorImpl::forward_for_single_feature_map(torch::Tensor &locations,
                                                                                       torch::Tensor &box_cls,
                                                                                       torch::Tensor &box_regression,
                                                                                       torch::Tensor &centerness,
                                                                                       const std::vector<std::pair<int64_t, int64_t>> &image_sizes)
{
    using namespace structures;
    auto N = box_cls.size(0), C = box_cls.size(1), H = box_cls.size(2), W = box_cls.size(3);

    box_cls = box_cls.view({N, C, H, W}).permute({0, 2, 3, 1});
    box_cls = box_cls.reshape({N, -1, C}).sigmoid();
    box_regression = box_regression.view({N, 4, H, W}).permute({0, 2, 3, 1});
    box_regression = box_regression.reshape({N, -1, 4});
    centerness = centerness.view({N, 1, H, W}).permute({0, 2, 3, 1});
    centerness = centerness.reshape({N, -1}).sigmoid();

    auto candidate_inds = box_cls > this->pre_nms_thresh;
    auto pre_nms_top_n = candidate_inds.view({N, -1}).sum(1);
    pre_nms_top_n = pre_nms_top_n.clamp(c10::nullopt, this->pre_nms_top_n);

    //todo change : find [:,:,none] how to use it in c++
    box_cls = box_cls * centerness.unsqueeze(2);

    std::vector<BoxList> results;
    for (int i = 0; i < N; i++)
    {
        auto per_box_cls = box_cls[i];
        auto per_candidate_inds = candidate_inds[i];
        per_box_cls = per_box_cls[per_candidate_inds];

        auto per_candidate_nonzeros = per_candidate_inds.nonzero();
        auto per_box_loc = per_candidate_nonzeros.select(1, 0);   //[:, 0];
        auto per_class = per_candidate_nonzeros.select(1, 0) + 1; //[:, 1] + 1;

        auto per_box_regression = box_regression[i];
        per_box_regression = per_box_regression[per_box_loc];
        auto per_locations = locations[per_box_loc];

        auto per_pre_nms_top_n = pre_nms_top_n[i];

        if (per_candidate_inds.sum().item().to<int>() > per_pre_nms_top_n.item().to<int>())
        {
            torch::Tensor top_k_indices;
            //todo check : if is it correct
            std::tie(per_box_cls, top_k_indices) = per_box_cls.topk(per_pre_nms_top_n.item().toInt(), -1, true, false);
            per_class = per_class[top_k_indices];
            per_box_regression = per_box_regression[top_k_indices];
            per_locations = per_locations[top_k_indices];
        }
        auto detections = torch::stack({
                                           per_locations.select(1, 0) - per_box_regression.select(1, 0),
                                           per_locations.select(1, 1) - per_box_regression.select(1, 1),
                                           per_locations.select(1, 0) - per_box_regression.select(1, 2),
                                           per_locations.select(1, 1) - per_box_regression.select(1, 3),
                                       },
                                       1);

        auto boxlist = BoxList(detections, image_sizes[i], "xyxy");
        boxlist.AddField("labels", per_class);
        boxlist.AddField("scores", per_box_cls);
        boxlist = boxlist.ClipToImage(false);
        boxlist = boxlist.RemoveSmallBoxes(this->min_size);
        results.push_back(boxlist);
    }

    return results;
}

std::vector<structures::BoxList> FCOSPostProcessorImpl::select_over_all_levels(const std::vector<structures::BoxList> &boxlists)
{
    using namespace structures;
    std::vector<BoxList> results;
    auto num_images = boxlists.size();
    for (size_t i = 0; i < num_images; ++i)
    {
        BoxList bList = boxlists[i];
        torch::Tensor scores = bList.GetField<>("scores");
        torch::Tensor labels = bList.GetField<>("labels");
        torch::Tensor boxes = bList.get_bbox();

        std::vector<BoxList> imgret;
        for (int j = 0; j < num_classes; j++)
        {
            auto inds = (labels == j).nonzero().view(-1);
            auto scores_j = scores.index(inds);
            auto boxes_j = boxes[inds].view({-1, 4});
            auto boxlist_for_class = BoxList(boxes_j, bList.get_size());
            boxlist_for_class.AddField("scores", scores_j);
            boxlist_for_class = boxlist_for_class.nms(nms_thresh);
            auto num_labels = boxlist_for_class.Length();
            torch::TensorOptions options;
            options.dtype = torch::kInt64;
            options.device = scores.get_device();
            //todo : check ```num_labels,```
            boxlist_for_class.AddField("labels", torch::full({
                                                                 num_labels,
                                                             },
                                                             j, options));

            imgret.push_back(boxlist_for_class);
        }

        auto result = BoxList::CatBoxList(imgret);
        auto number_of_detections = imgret.size();

        if (number_of_detections > fpn_post_num_top_n && fpn_post_num_top_n > 0)
        {
            auto cls_scores = result.GetField("score");
            auto image_thresh = torch::kthvalue(cls_scores.cpu(), number_of_detections - fpn_post_num_top_n + 1);
            auto keep = cls_scores >= std::get<0>(image_thresh).item();
            keep = torch::nonzero(keep).squeeze(1);
            result = result[keep];
        }

        results.push_back(result);
    }

    return results;
}

FCOSPostProcessor MakeFCOSPostprocessor(float pre_nms_thresh, int pre_nms_top_n,
                                        float nms_thresh, int fpn_post_num_top_n, int min_size,
                                        int num_classes)
{
    return FCOSPostProcessor(pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_num_top_n, min_size, num_classes);
}

} // namespace modeling
} // namespace rcnn