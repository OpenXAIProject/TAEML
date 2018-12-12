# TAEML (Few-shot Classificaiton)

### **CONTENT**
> TAEML: Task-Adaptive Ensemble of Meta-Leaners for Few-shot Classification
### **How to Use**

# Uncetainty-Aware Attention for Reliable Interpretation and Prediction
+ Jay Heo(KAIST, Co-author), Hae Beom Lee (KAIST, Co-author), Saehoon Kim (AITRICS), Juho Lee (Univ. Oxford), Kwang Joon Kim(Yonsei University College of Medicine), Eunho Yang (KAIST), and Sung Ju Hwang (KAIST)

<b> Update (November 4, 2018)</b> TensorFlow implementation of [Uncetainty-Aware Attention for Reliable Interpretation and Prediction](https://arxiv.org/pdf/1805.09653.pdf) which introduces uncertainty-aware attention mechanism for time-series data (in Healthcare). We model the attention weights as Gaussian distribution with input dependent noise that the model generates attention with small variance when it is confident about the contribution of the gived features and allocates noisy attentions with large variance to uncertainty features for each input.

## Abstract
<p align="center">
<img width="633" height="391" src="https://github.com/OpenXAIProject/UncertintyAttention_DropMax/blob/master/UA-master/ua_model.png">
    </p>
Attention mechanism is effective in both focusing the deep learning models on relevant features and interpreting them. However, attentions may be unreliable since the networks that generate them are often trained in a weakly-supervised manner. To overcome this limitation, we introduce the notion of input-dependent uncertainty to the attention mechanism, such that it generates attention for each feature with varying degrees of noise based on the given input, to learn larger variance on instances it is uncertain about. We learn this Uncertainty-aware Attention (UA) mechanism using variational inference, and validate it on various risk prediction tasks from electronic health records on which our model significantly outperforms existing attention models. The analysis of the learned attentions shows that our model generates attentions that comply with clinicians’ interpretation, and provide richer interpretation via learned variance. Further evaluation of both the accuracy of the uncertainty calibration and the prediction performance with “I don’t know” decision show that UA yields networks with high reliability as well.




<img src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width="300" height="100">

# XAI Project

### **Project Name**
> A machine learning and statistical inference framework for explainable artificial intelligence(의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)
### **Managed by**
> Ministry of Science and ICT/XAIC
### **Participated Affiliation**
> UNIST, Korean Univ., Yonsei Univ., KAIST., AITRICS
### **Web Site**
> <http://openXai.org>
