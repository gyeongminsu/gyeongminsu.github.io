---
title : BERT 논문리뷰
categories : ML DL BERT Transformer Paper-Review
tags : ML DL BERT Transformer Paper-Review
date : 2023-09-28 18:00:00 +0900
pin : true
path : true
math : true
image : /assets/img/2023-09-28-BERT-review/thumbnail.png
---

# BERT 논문리뷰

# 1. 개요

Language Model(LM)의 pre-training 방법은 기존의 많은 NLP 태스크에서 효율적으로 개선되어 굉장히 좋은 성능을 내고 있다. 

이는 특히 <span style="color:#BA6835">전체적인 sentenct(문장) 간의 관계를 예측하는 sentence(문장) 단위 레벨의 태스크(ex. Nature Language Inference)</span>를 목표하여 발전되어 왔다.

또한 <span style="color:#BA6835">Token(토큰) 단위의 태스크(ex. NER, QA)</span>에서도 많은 발전을 이루었다. 이러한 토큰 단위 태스크를 진행하는 모델은 토큰 단위의 fine-grained output을 만들어야 한다.

- fine-grained output : 하나의 output을 내기 위해 작은 단위의 output process로 나눈 후 수행하는 것.

Pre-training을 진행한 모델을 세부 목표 태스크인 downstream task에 적용하기 위한 두 가지 방법이 존재한다.

1. feature based approach
    
    ex) ELMo
    
    ELMo는 pre-trained representations를 하나의 추가적인 feature로 활용해 task-specific 아키텍처를 사용한다.
    
2. fine-tuning approach
    
    ex) GPT
    
    GPT는 downstream task-specific(fine-tuned) parameter의 수는 최소화하고, 모든 pre-trained parameter를 조금만 바꿔서 task-specific하게 학습한다.
    

pre-trained된 모든 parameter를 간단히 미세 조정하여 downstream task를 시행한다.

pre-training 과정에서 feature based와 fine-tuning 두 과정 모두 같은 object function을 공유한다. 이 과정에서 일반적으로 language representation을 학습하기 위해 단방향적 언어 모델(unidirectional language model)을 사용하게 된다.

<span style="color:#BA6835">unidirectional model을 fine-tuning할 때 생기는 아키텍쳐의 제한은 모델의 pre-trained representation의 성능을 떨어뜨린다.</span>

GPT의 경우, 입력된 문장의 토큰이 이전 토큰과의 self-attention score만 계산하게 된다.

이러한 제한은 문장(sentence) 수준의 태스크에서 optimal(최적)하지 않다. 

문장 수준의 태스크에서 optimal한 태스크는 <span style="color:#BA6835">양방향에서 attention score를 계산하는 것이다.</span>

fine-tuning approach를 개선시킨 BERT : Bidirectional Encoder Representations from Transformers는 pre-training 단계에서 “masked language model(MLM)” 방법을 사용하여 unidirectional model의 제약을 완화하였다.

## 1-1. WordPiece Embedding

BERT는 입력된 문장을 tokenize할 때 <span style="color:#BA6835">단어보다 더 작은 단위로 쪼개는 subword tokenizer의 종류 중 하나인 WordPiece Tokenizer를 사용한다.</span> 

subword tokenizer는 기본적으로 자주 등장하는 단어는 그대로 단어 집합에 추가하지만, 자주 등장하지 않는 단어의 경우에는 더 작은 단위인 subword로 분리하여subword들이 단어 집합에 추가된다는 아이디어를 갖고 있다. 이렇게 단어 집합이 만들어지고 나면, 이 단어 집합을 기반으로 tokenizing을 진행한다.

wordpiece embedding을 진행하여 BERT는 wordpiece embedding layer를 보유한다.

## 1-2. Position Embedding

BERT는 sentence 안에서 단어 token의 위치 정보를 embedding하기 위해 Position Embedding이라는 방법을 사용하였다.

트랜스포머에서는 Positional Encoding이라는 방법을 통해서 token의 위치 정보를 embedding했다. 포지셔널 인코딩은 sin함수와 cos함수를 사용하여 token의 위치에 따라 다른 값을 가지는 행렬을 만들어 이를 단어 벡터들과 더하는 방법이다.

 BERT에서는 이와 유사하지만, <span style="color:#BA6835">위치 정보를 sin함수와 cos함수로 만드는 것이 아니라 학습을 통해서 얻는 Position Embedding이라는 방법을 사용한다.</span>

![Untitled](/assets/img/2023-09-28-BERT-review/Untitled.png)

위의 그림은 Position Embedding을 사용하는 방법을 보여준다. 

위의 그림에서 WordPiece Embedding은 input sentence를 조작하지 않고 embedding한 것이다. 그리고 이 입력에 position embedding을 통해서 위치 정보를 더해주어야 한다. 

position embedding에서는 위치 정보를 위한 embedding layer를 하나 더 사용한다. 

ex) 문장의 길이가 4라면 4개의 position embedding vector를 학습시킨다. 그리고 BERT의 입력마다 다음과 같이 position embedding vector를 더해주는 것이다.

- 첫번째 단어의 embedding vector + 0번 position embedding vector
- 두번째 단어의 embedding vector + 1번 position embedding vector
- 세번째 단어의 embedding vector + 2번 position embedding vector
- 네번째 단어의 embedding vector + 3번 position embedding vector

실제 BERT에서는 문장의 최대 길이를 512로 하고 있다. 따라서 총 512개의 position embedding vector가 학습된다. 

이렇게 학습한 position embedding vector로 BERT는 position embedding layer를 보유하게 된다.

## 1-3. Segment Embedding

BERT는 두 개의 문장 입력이 필요한 태스크를 수행하기도 한다.

앞서 설명한 두 개의 embedding layer에 더불어 BERT는 입력된 두 개의 문장에서 첫 번째 문장에는 Sentence 0 embedding, 두 번째 문장에는 Sentence 1 embedding을 더해주어 두 개의 embedding vector를 사용하여  Segment Embedding layer를 가지어 사용하게 된다.

# 2. Architecture of BERT

BERT의 기본 구조는 <span style="color:#BA6835">Bidirectional Transformer의 Encoder를 쌓아올린 구조</span>이다. BERT-Base Version에서는 총 12개, BERT-Large Version에서는 총 24개를 쌓아올렸다. 

Transformer-Encoder layer의 수 : L, d_model의 크기 : D, self-attention head의 수 : A 라고 할 때, BERT-Base와 BERT-Large의 파라미터 크기는 다음과 같다.

- BERT-Base : L=12, D=768, A=12 : 110M of Parameters
- BERT-Large : L=24, D=1024, A=16 : 340M of Parameters

Transformer model의 하이퍼파라미터가 L=6, D=512, A=8인 것을 생각했을 때 BERT model은 Transformer보다 확실히 규모가 크다는 것을 알 수 있다. 

BERT-Base는 GPT-1과 하이퍼파라미터의 수가 동일하다. 이 이유는 BERT를 설계할 때 직접적으로 GPT-1과 성능을 비교하기 위해 동등한 크기로 BERT-Base를 만들었기 때문이다.

BERT-Large는 BERT의 최대 성능을 보여주기 위해 파라미터를 늘리어 만든 모델이다.

![Untitled](/assets/img/2023-09-28-BERT-review/Untitled%201.png)

BERT는 input sentence를 위에서 설명한 WordPiece, Segment, Position 세 가지의 Embedding 을 거쳐 만들어진 Embedding vector를 합치어 encoder의input sequence로 가공한다.

모든 input sequence의 첫 번째 토큰은 [CLS] 토큰으로 시작한다. 그리고 input sentence에 여러 문장이 입력되면 [SEP] 토큰으로 문장을 구분한다.

BERT의 input representation은 이렇게 생성한 토큰들(segment, position, wordpiece)을 모두 합해준 것이다.

# 3. Pre-training of BERT

![Untitled](/assets/img/2023-09-28-BERT-review/Untitled%202.png)

BERT, GPT, ELMo의 아키텍처를 살펴보면 ELMo 또한 bidirectional(양방향성) model이라고 생각할 수 있다. 

하지만 ELMo는 정방향 LSTM과 역방향 LSTM 모델을 각각 훈련시켜 마지막에 병합하는 형식이기 때문에 한 개의 모델 내부에서 bidirectional(양방향성)한 학습이 진행되지 않는다.

개요에서 설명했듯이 BERT는 pre-training 과정에서 MLM을 통해 bidirectional의 특성을 얻게 되었다.

이어서 BERT가 pre-training을 진행하는 두 가지 방법인 Masked Language Model과 Next Sentence Prediction에 대한 설명을 시작하겠다.

## 3-1. Masked Language Model

Masked Language Model(MLM)은 입력 텍스트 단어 토큰의 15%를 랜덤으로 마스킹(Masking)하는 모델이다. 여기서 마스킹이란 원래의 단어가 무엇이였는지 모르게 하는 것이다. 그리고 입력 텍스트 사이의 마스킹되어있는 토큰들을 문맥에 근거하여 모델이 예측하도록 한다.

여기서 선택된 15%의 토큰은 사실 전부 마스킹되는 것은 아니다. 선택된 토큰 사이에서 아래의 세 가지 규칙이 적용된다.

1. 80%의 토큰은 [MASK]로 변경된다.
2. 10%의 토큰은 랜덤하게 다른 단어로 변경된다.
3. 10%의 토큰은 그대로 둔다.

세 가지 규칙을 적용하는 이유는, <span style="color:#BA6835">선택된 토큰들을 모두 마스킹할 경우  마스킹된 [MASK] 토큰은 fine-tuning 단계에서 나타나지 않으므로 pre-training 단계와 fine-tuning 단계에서의 불일치가 발생하는 문제가 생기기 때문이다.</span> 따라서 선택된 모든 단어에 마스킹을 적용하지 않는 것이다.

이와 같은 방법으로 cross entropy loss function을 이용해 [MASK] 토큰에 대한 예측을 수행한다.

## 3-2. Next sentence prediction(NSP, 다음 문장 예측)

BERT는 input으로 2개의 sentence가 입력되어 이 문장이 이어지는 문장인지 아닌지를 맞추는 방식으로 pre-training이 진행된다. 

training의 input으로 실제로 이어지는 문장과 랜덤으로 이어붙인 문장이 50:50의 비율로 주어진다.

input sentence의 문장의 연속성을 확인한 후 이어지는 문장이면 IsNextSentence, 이어지지 않으면 NotNextSentence로 labeling한다.

![Untitled](/assets/img/2023-09-28-BERT-review/Untitled%203.png)

BERT의 Pre-training, Fine-Tuning을 진행하는 과정을 표현한 위의 그림에서 BERT 박스 안의 좌측 상단에 있는 C token을 이용하여 input sentence의 두 문장의 원본 corpus 말뭉치에서 이어지는 문장인지 아닌지에 대해 학습을 진행한다.

BERT는 위키피디아(25억 단어)와 BooksCorpus(8억 단어)와 같은 레이블이 없는 텍스트 데이터를 이용하여 위에서 설명한 pre-training의 과정을 진행하였다. 

BERT는 총합 33억 개의 단어 Corpus에 대해 40 epochs씩 학습하여 대략 100만 step의 training을 진행하였다.

pre-training 과정의 세부 스펙은 아래와 같다.

- $10^{-4}$의 학습률로 Adam Optimizer
- 0.01 학습률로 L2 Normalization
- 0.1 비율 dropout
- Activate function : gelu
- Batch size : 256

# 4. Fine-Tuning of BERT

BERT가 높은 성능을 얻을 수 있었던 것은 <span style="color:#BA6835">레이블이 없는 방대한 데이터를 이용한 pre-trained model을 downstream task에서 레이블이 있는 데이터를 이용한 추가 training하여 하이퍼파라미터 재조정을 진행하였기 때문이다.</span>

이러한 downstream task에서 추가 training을 거쳐 하이퍼파라미터 재조정하는 과정을 Fine-Tuning이라고 한다.

BERT는 Transformer 모델을 쌓아올려 만든 모델이므로 self-attention 메커니즘을 이용하여 fine-tuning을 진행할 수 있다.

BERT를 Fine-Tuning하여 수행하는 downstream task의 종류는 다음과 같다.

1. **Single Text Classification - 단일 sentence**
input sentence로 단일 문서가 입력되었을 때 분류 학습을 진행한다. 
각 sentence를 구분하는 [CLS] 토큰 위치의 output layer에 FC layer 등을 추가하여 분류 예측을 진행한다.
2. **Tagging - 단일 sentence**
품사 태깅, 개체명 인식 등의 태깅 작업에서 BERT를 사용할 수 있다.
출력층의 각 토큰마다 Dense layer를 추가하여 분류 예측을 수행한다.
3. **Text Pair Classification&Regression - 두 개의 sentence**
input sentence로 두 문장을 입력받았을 때 [SEP] token을 이용하여 두 문장 간의 논리 관계를 예측한다. ****
4. **Question Answering  - 두 개의 sentence**
질문 sentence, 본문 sentenece 쌍을 입력받아 출력층에 추가한 Dense layer에서 대답을 예측하도록 한다.
****