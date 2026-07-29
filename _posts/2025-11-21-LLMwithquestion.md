---
title : 언어 모델과 그를 둘러싼 질문들
categories : [LLM, Kuhn, NormalScience, Nietzsche, Linguistics]
tags : [LLM, Kuhn, NormalScience, Nietzsche, Linguistics]
date : 2025-11-21 18:00:00 +0900
pin : true
path : true
math : true
image : /assets/img/2025-11-21-LLMwithquestion/thumbnail.jpg
toc : true
layout : post
comments : true
---

# 0. Introduction

후배들이 많이 활동하는 학과 소모임에서 선배로서 지식 공유회 발표를 해달라는 요청이 왔다. 현재 자연어처리 연구실에 있으니 트랜스포머에 기반한 대형 언어 모델 연구에 관한 발표를 목적으로 요청을 했겠지만, 식상한 공학 연구 얘기는 하기가 싫었다.

연구실에서의 연구 외에 관심을 갖고 많이 탐구하고 있는 철학 주제만 얘기하자니 아직 초심자의 단계라 정리되지 않은 상태에서 발표하기엔 내용의 갈피가 쉽게 잡히지 않았다.

결국 나에게도, 학과 후배들에게도 익숙한 언어 모델에 대한 내용을 중심으로 언어학, 과학철학, 대륙철학에 손가락을 찍어 맛만 보는 형식의 발표 내용을 준비하게 되었다.

이제 발표 슬라이드와 슬라이드 내용에 관한 간단한 첨언들을 시작하겠다.

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0001.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0002.jpg)

# 1. 우리가 언어를 배우는 방법

스키너의 행동주의 심리학 및 언어학이 등장하기 전 언어학은 구조주의 철학과 연계된 매우 사변적인 학문이였다. 스키너가 행동주의에 기반한 심리학을 제창함으로써 언어학은 경험주의의 색채가 입혀지기 시작했다. 따라서 언어학에 대한 소개를 스키너의 행동주의부터 시작하여 촘스키의 생성문법, 그리고 현재 가장 논의가 활발한 사용 기반 언어학, 인지 언어학으로의 흐름으로 연결했다.

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0004.jpg)



![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0005.jpg)


        
![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0006.jpg)
        
# 2. 언어 모델링의 발전

컴퓨터가 발명된 이후 기존에 파헤쳐졌던 언어학을 기반으로 하여 언어의 분류 및 생성을 논리만으로 시뮬레이션하려는 시도가 시작되었다. 언어 모델링의 시초에는 통계학을 이용하지 않은 규칙 기반의 모델링만을 하였으나, 언어의 모든 부분을 Rule-base로 진행하려니 너무나 많은 변수와 한계에 부딪혀 통계적 모델링을 시작하였다. 결국 통계학에 기반한 언어 모델은 지금 가장 큰 영향력을 행사하고 있고 꽤나 그럴듯하게 인간의 언어를 시뮬레이션하고 있다.

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0008.jpg)


![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0009.jpg)

트랜스포머 기반의 언어 모델이 대세가 되고 계속 발전되어 OpenAI가 GPT 시리즈를 개발했을 때 Few-shot Learning, In-Context Learning에 기반한 창발적 능력은 연구자들 사이에 엄청난 화두가 되었다. 


![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0010.jpg)

프롬프트에 언어 모델이 풀려는 태스크와 유사한 질의응답 예시를 같이 입력해주면 성능이 올라간다는 Few-shot Learning은 GPT-3 부터 창발하게 되었고, 언어 모델의 파라미터 규모가 커짐에 따라 디자인하지 않았던 능력이 생긴 것이므로 공간 단위의 창발이라고 할 수 있겠다.

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0011.jpg)

grokking 또한 언어 모델의 연구자들이 예상하거나 의도하지 않은 창발이라고 볼 수 있다. 우연히 모델 훈련을 종료하지 않은 채 퇴근했던 연구자들이 다음날 확인해보니 모델의 테스트셋에 대한 일반화 능력이 어느 순간 불연속적으로 상승하게 됐고, 이것은 시간 단위의 창발이라고 할 수 있다.

# 3. 생각해볼 수 있는 것들

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0013.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0014.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0015.jpg)

[Learning without training: The implicit dynamic of in-context learning](https://arxiv.org/abs/2507.16003) 논문과 [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2510.15511) 논문은 각각 비교적 매우 최신에 나온 논문들이고 각 실험 디자인 및 결과, 그리고 굉장히 엄밀한 수학적 증명을 제시한 논문들이다. 



하지만 해당 논문들에서는 각각 단일 Transformer block과 gpt-2, llama3.1-8b와 같이 현재에 비하면 outdate되거나 매우 작은 모델만을 이용했고, 후자의 논문에서 임베딩 공간 상에서의 일대일 대응을 증명하기 위해 충돌을 일으키는 파라미터 집합이 0이라는 것을 르베그 측도가 0이라는 것으로 보였는데 이것은 단일 트랜스포머 모델 상에서만 수학적 증명을 진행하였다. 따라서 트랜스포머 블럭이 쌓이거나 파라미터 규모가 커진다면 어떻게 달라질지는 미지수로 남아있다.

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0016.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0017.jpg)

Grokking에서 언어 모델의 Test set에 대한 성능이 갑자기 오르는 것은 마치 언어 모델이 테스트 셋을 풀기 위한 법칙을 깨달은 것처럼 느껴진다. 앞서 제시된 사항들은 사용 기반 언어학의 손을 들어주더라도, Grokking은 촘스키의 언어 본능을 지지해주는 마지막 하나의 단서가 될 수도 있지 않을까?

# 4. 우리가 고민해봐야 할 것들

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0019.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0020.jpg)

요즘 시대에 특히 대학생, 대학원생, 직장인이라면 LLM을 안 써본 사람이 매우 적을 것이다. 우리의 삶에서 LLM을 이용하는 비중은 점점 늘어나고 있고, 이제는 없었던 때의 삶을 상상하거나 되돌리기 힘들 정도이다. 하지만 이렇게 LLM에 의존하면서 살아도 되는지 의심하는 사람들도 많을 것이다.

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0021.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0022.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0023.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0024.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0025.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0026.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0027.jpg)

![image.jpg](/assets/img/2025-11-21-LLMwithquestion/jpg-0028.jpg)