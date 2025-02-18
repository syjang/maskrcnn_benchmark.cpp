#include "rpn/rpn.h"
#include <iostream>

#include <defaults.h>


namespace rcnn{
namespace modeling{

RPNHeadImpl::RPNHeadImpl(int64_t in_channels, int64_t num_anchors)
  :conv_(register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1)))),
  cls_logits_(register_module("cls_logits", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_anchors, 1)))),
  bbox_pred_(register_module("bbox_pred", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_anchors * 4, 1))))
{
  for(auto &param : named_parameters()){
    if(param.key().find("weight") != std::string::npos) {
      torch::nn::init::normal_(param.value(), 0, 0.01);
    }
    else{
      torch::nn::init::zeros_(param.value());
    }
  }
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> RPNHeadImpl::forward(std::vector<torch::Tensor>& x){
  std::vector<torch::Tensor> logits, bbox_reg;
  logits.reserve(x.size());
  bbox_reg.reserve(x.size());
  torch::Tensor t;
  for(auto& feature: x){
    t = conv_->forward(feature).relu_();
    logits.push_back(cls_logits_->forward(t));
    bbox_reg.push_back(bbox_pred_->forward(t));
  }
  return std::make_pair(logits, bbox_reg);
}

RPNModuleImpl::RPNModuleImpl(int64_t in_channels)
  :anchor_generator_(register_module("anchor_generator", MakeAnchorGenerator())),
  head_(register_module("rpnhead", RPNHead(in_channels, anchor_generator_->NumAnchorsPerLocation()[0]))),
  rpn_box_coder_(BoxCoder(std::vector<float>{1.0, 1.0, 1.0, 1.0})),
  box_selector_train_(register_module("box_selector_train", MakeRPNPostprocessor(rpn_box_coder_, /*is_train=*/true))),
  box_selector_test_(register_module("box_selector_test", MakeRPNPostprocessor(rpn_box_coder_, /*is_train=*/false))),
  loss_evaluator_(MakeRPNLossEvaluator(rpn_box_coder_)),
  rpn_only_(rcnn::config::GetCFG<bool>({"MODEL", "RPN_ONLY"})),
  in_channels_(in_channels){}

std::shared_ptr<RPNModuleImpl> RPNModuleImpl::clone(torch::optional<torch::Device> device) const{
  torch::NoGradGuard no_grad;
  std::shared_ptr<RPNModuleImpl> copy = std::make_shared<RPNModuleImpl>(in_channels_);
  auto named_params = named_parameters();
  auto named_bufs = named_buffers();
  for(auto& i : copy->named_parameters()){
    i.value().copy_(named_params[i.key()]);
  }
  for(auto& i : copy->named_buffers()){
    i.value().copy_(named_bufs[i.key()]);
  }
  if(device.has_value())
    copy->to(device.value());
  return copy;
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward(rcnn::structures::ImageList& images, std::vector<torch::Tensor>& features, std::vector<rcnn::structures::BoxList> targets){
  //given targets
  std::vector<torch::Tensor> objectness, rpn_box_regression;
  std::tie(objectness, rpn_box_regression) = head_->forward(features);
  std::vector<std::vector<rcnn::structures::BoxList>> anchors = anchor_generator_(images, features);
  
  if(is_training()){
    return forward_train(anchors, objectness, rpn_box_regression, targets);
  }
  else{
    return forward_test(anchors, objectness, rpn_box_regression);
  }
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward(rcnn::structures::ImageList& images, std::vector<torch::Tensor>& features){
  //no targets
  std::vector<torch::Tensor> objectness, rpn_box_regression;
  std::tie(objectness, rpn_box_regression) = head_->forward(features);

  std::vector<std::vector<rcnn::structures::BoxList>> anchors = anchor_generator_(images, features);
  
  return forward_test(anchors, objectness, rpn_box_regression);
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward_train(std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& rpn_box_regression, std::vector<rcnn::structures::BoxList> targets){
  std::vector<rcnn::structures::BoxList> boxes;
  boxes.reserve(anchors.size());
  torch::Tensor loss_objectness, loss_rpn_box_reg;
  std::map<std::string, torch::Tensor> losses;
  if(rpn_only_){
    //cat anchors per image [not in original implementation]
    for(int i = 0; i < anchors.size(); ++i)
      boxes.push_back(rcnn::structures::BoxList::CatBoxList(anchors[i]));
  }
  else{
    //no_grad bracket
    {
      torch::NoGradGuard guard;
      boxes = box_selector_train_->forward(
        anchors, objectness, rpn_box_regression, targets
      );
    }
  }
  std::tie(loss_objectness, loss_rpn_box_reg) = loss_evaluator_(anchors, objectness, rpn_box_regression, targets);
  losses["loss_objectness"] = loss_objectness;
  losses["loss_rpn_box_reg"] = loss_rpn_box_reg;
  return std::make_pair(boxes, losses);
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward_test(std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& rpn_box_regression){
  std::vector<rcnn::structures::BoxList> boxes = box_selector_test_->forward(anchors, objectness, rpn_box_regression);
  std::map<std::string, torch::Tensor> losses;
  if(rpn_only_){
    for(auto& box: boxes){
      //get index and sort box
      box = box[std::get<1>(box.GetField("objectness").sort(/*dim=*/-1, true))];
    }
  }
  return std::make_pair(boxes, losses);
}

RPNModule BuildRPN(int64_t in_channels){
  return RPNModule(in_channels);
}

}
}