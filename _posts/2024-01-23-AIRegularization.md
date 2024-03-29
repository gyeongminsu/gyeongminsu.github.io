---
title : AI의 발전 방향성과 AI 규제에 대한 고찰
categories : ML DL XAI AI_Regularization
tags : ML DL XAI AI_Regularization
date : 2024-01-23 12:00:00 +0900
pin : true
path : true
math : true
image : /assets/img/2024-01-23-AIRegularization/thumbnail.png
toc : true
layout : post
comments : true
---

# 1. AI principle 만들기 (Algorithmic Bias / Disinformation 등을 어떻게 줄이고 극복할 수 있는가?)

Google, Microsoft, OpenAI 등등 현재 IT의 최전선을 달리고 있는 기업들은 각 사에서 제
정한 AI principle이 존재한다. 또한 국제 기구 중 최고의 권위를 갖고 있는 OECD에서도 AI
principle에 대한 policy를 계속 제정하려는 시도를 보이고 있다.

AI principle을 제정하려는 목적에는, AI의 설계 목적에 어긋난 응용을 최소화하고 AI의 발전
에 따른 혜택이 전 세계의 사람들에게 골고루 와닿게 하기 위함(비차별적, Political Correct)
이 제일 클 것이다.

OpenAI에서 2023년 3월에 GPT-4를 발표하면서 발행한 GPT-4 Technical Report에서 다
룬 윤리적 문제를 살펴보면,

1. Hallucination : 말이 안 되거나 믿을 수 없는 정보
2. Harmful content : 선정적, 폭력적 내용
3. Harms of representation, allocation, and quality of sevice : 혐오 표현
4. Disinformation and influence operations : 허위 정보와 사회적 영향
5. Proliferation of conventional and unconventional weapons : 신/구 무기의 확산
6. Privacy : 개인 정보
7. Cybersecurity : 사이버 보안
8. Potential for risky emergent behaviors : 잠재적으로 위험한 부작용
9. Interactions with Other Systems : 다른 시스템과의 상호작용
10. Economic impacts : 경제적 영향력
11. Acceleration : 경쟁 가속화
12. Overreliance : 과의존

총 12개의 항목이 소개되고 있다.

<span style="color:#BA6835">기존의 설계와 달리 특정한 feature에 더 무게를 두는 Algorithmic bias</span>와 <span style="color:#BA6835">할루시네이션 등
의 잘못되고 허황된 정보(disinformation)</span>는 요즘 LLM을 이용한 생성형 AI가 발전하면서 가
장 크게 대두되고 있는 문제들이다.

새로운 AI 모델을 개발할 때, 모델이 추론(inference)을 통하여 내놓는 결과가 한 방향으로
치우지지 않게끔 하기 위하여 연구자들은 Inductive Bias를 주입한다. 이것은 <U>연구자들이
모델에 의도적으로 주입하는 편향</U>이라고 볼 수 있는데, <span style="color:#BA6835">모델이 사전 훈련(pre-training) 단계
에서는 접하지 못한 데이터에 대하여 올바른 output을 산출하게 하기 위하여 구상하는 것</span>이
다.

AI 모델에서 Algorithmic bias는 초기 설계와 사전 훈련 단계에서 적절한 훈련을 진행하지
못 하여 발생하는 경우가 많다.

이는 어찌보면 인류에게 AI가 현재 과도기적인 상황에 놓여있고, 모델의 규모에 비해 너무
많은 범위에서 사용하려고 함으로써 발생하는 오류라고도 할 수 있다.

AI의 뜻을 풀어 헤쳐 보면, 인류의 지능을 모방하여 인공적으로 만들어서 본뜬 시스템이라고
할 수 있다.

그러나, <span style="color:#BA6835">인류는 아직 인류에 대해 완벽히 알지 못한다. 뇌의 작동기전, 인지심리학 등의 연구
분야는 아직 밝혀지지 않은 것이 너무나도 많고, 쏟아지는 새로운 정보가 엄청나게 많다.</span>

결국 Algorithmic bias나 disinformation같은 부작용은 지구 생태계와 인류에 대한 근본적
인 이해가 진행됨으로써, 진일보한 새로운 모델을 개발함으로써 해결할 수 있을 것이다.

LLM에 Ontology와 같은 정보 표현 방식을 결합하여 LLM이 가진 정보 체계의 정교성을 높
이려는 연구도 많이 진행되고 있다.

이렇게 <U>AI의 알고리즘을 발전시켜 부작용을 없애기 위해, 자연과학의 발전이 우선시되어야
한다. 기술의 발전은 늘 자연과학을 통한 발견을 응용하여 공학의 발전이 진행되는 선후 관
계가 존재해왔다.</U>

이러한 관점에서 기초과학에 대한 투자를 늘리고 연구 규모를 넓힘으로써, 인류의 삶을 더
공평하고 풍요롭게 만들 수 있을 것이다.

# 2. AI Safety / Regulation 관련 어떤 방식으로 접근하고 미리 대비해야 앞으로 AI 요소가 들어갈 수 있는 다양한 분야에서 Safety 관련 미리 대비하고 적용할 수 있을지

기술의 발전에 따라서 AI 모델의 규모가 점점 커지고 있고, 규모가 커짐에 따라 AI 모델이 수
행할 수 있는 업무 범위 또한 계속 늘어나고 있다.

그러면서 <U>AI 모델이 처음 설계될 때의 목적과 다른 분야에서 성능을 보이는</U> <span style="color:#BA6835">창발적 능력
(emergent ability)</span>이라는 개념이 대두되고 있다. 특히 생성형 AI가 대세가 되고, 여러 산업
도메인의 업무에 AI가 응용되면서 창발적 능력은 더욱 조명을 받을 것이다.

예상치 못한 분야에서 뜻밖의 능력을 보여주는 것이 창발적 능력이기 때문에, 이러한 상황에
대한 대비책을 완벽하게 세우는 것은 힘들더라도 어느 정도 정교하게 준비하는 과정은 필요
하다.

불의 발견과 사용은 인류사의 시작부터 지금까지 계속되어 왔고, 어찌보면 인류가 가장 최초
로 사용한 기술이기도 하지만 요즘 시대에도 화재로 인한 사고는 끊이지 않는다. 기술의 발
전에 따라 우리의 삶은 풍요로워짐과 동시에, 막을 수 없는 부작용도 생기기 마련이다.

이렇듯 AI에 관한 부작용을 완벽히 억제하는 것은 어려울 것이다. 다만, 이렇게 정교
하게 발전된 AI를 안전하게 이용하려는 노력은 분명 계속해야 한다.

시간이 흐르면서 AI가 점점 더 다양한 도메인에 응용되고 있지만, 각 도메인에서 정한 목표
에 알맞게 사용하려면 AI 모델의 결정 과정에 대한 투명성이 보장되어야 한다.

모델의 투명성을 보장하기 위해 XAI 라는 주제의 연구가 활발히 시도되고 있다. XAI는
Explainable AI의 약자로, 설명 가능한 인공지능이라는 뜻이다.

심층 신경망을 이용하여 딥러닝을 진행하는 모델은 <U>블랙박스 모델</U>이라고 불리는데, 이는 모
델에 주어진 입력(input)에 대한 결과(output)의 산출 과정이 불투명하기 때문이다. XAI의
연구는 이러한 <U>불투명성을 투명화하기 위한 노력</U>이라고 볼 수 있다.

블랙박스 모델의 인풋에 대한 계산 과정에서 어떠한 지표를 세워서 어떤 기준을 이유로 모델
의 계산 과정을 가시화하여 설명하는 것을 XAI라고 말할 수 있는데, 최근 XAI 연구의 트렌드
는 <span style="color:#BA6835">모델의 학습이 이루어지고 난 뒤 설명을 진행하는 Post-hoc explanation</span>이다. 모델에 대
해서 사후 설명해야 한다는 과정상의 특징으로 볼 때 학습에 사용된 알고리즘에 구애받지 않
는, Model-agnostic explanation의 연구가 활발하다.

특히, LLM의 골조가 되는 트랜스포머 모델의 attention 계산에 대한 explanation의 연구도
활발히 진행되는 중이다.

이렇게 <span style="color:#BA6835">블랙박스 모델의 계산 과정에서 올바르고 공통된 지표가 정립되어야, 각 도메인의 특
수한 상황에서 알맞게 AI 모델을 이용할 수 있을 것이다. 그래야 AI 모델의 오용을 방지할 수
있고, 부작용도 최소화할 수 있다.</span>

AI의 Safety에 관한 규제 또한 이러한 설명 가능한 지표를 이용해서 마련할 수 있을 것이다.

# 3. 다양한 업무, 다양한 주제(인권, 환경, 교육, 평등) 그리고 창업 등에 AI를 어떻게 적용하고, 그 안에서 Safety(Bias) 관련 어떻게 대비해야 하는지

현재 개발되고 있는 AI 모델은, 근본적으로 주어진 데이터를 학습하여 그에 걸맞는 결과를
산출하는 방식으로 작동한다.

가장 간단하게 설명하자면, AI에게 다양한 주제에 맞는 특별한 데이터를 입력하여 학습시킨
후, 그 도메인에 걸맞는 결과를 산출하도록 하는 것이 다양한 주제에 대한 AI의 적용 방법이
라고 할 수 있겠다.

예전에 러시아와 미국이 합작을 하여 만든 우주 로켓이 발사하자마자 폭발하는 사건이 있었
는데, 그 이유는 러시아 과학자는 길이의 기본 단위로 인치를 이용하고, 미국의 과학자는 센
티미터를 사용하였기 때문이라고 한다.

우리는 협업을 할 때 소통을 해야만 하고, 소통 시 의견의 오차를 줄이기 위해 도량형을 제정
하고, 법규를 만들었다.

AI는 결국 우리의 도구로 사용되기 때문에, 거시적인 principle을 세우는 것도 중요하지만 <U>각
주제에 사용하기에 앞서 주제에 특화된 베이스라인을 정하는 것</U>이 급선무이다.

이러한 베이스라인을 명확히 정하기 위해 AI 전문가와 각 도메인 전문가들의 커뮤니케이션
기회를 늘리고, 각자의 상황에 대한 이해를 깊게 시키는 것이 필요할 것이다.

# 4. AI의 트렌드, AI의 다음 단계나 떠오르는 AI 분야는 무엇인지

자연과학의 발전에 맞춰서 기초과학의 진보된 내용을 응용하는 새로운 인공지능이 등장할
것이다.

시간의 흐름에 따라 컴퓨팅 자원 능력의 향상, 데이터 양의 증가, 파라미터 수의 증가 등으로
인해 AI 모델의 규모는 점점 커질 것이고, 기존에는 숙달된 전문가만이 수행할 수 있는 능력
을 AI가 대체하게 되는 날이 올 것이다.

인류의 업무에서 인류가 주된 진행을 하고, AI를 보조 도구로 이용하게 될 것이다.

양자컴퓨팅의 발전에 따라 새로워진 규격에 맞는 기계가 등장할 수 있고, 기존 컴퓨팅의 패
러다임 자체가 변할 수 있다.

신체 기능의 일부가 손상된 사람들에게, 우리 신체의 기능을 모방한 기계를 이용함으로써 건
강한 삶을 살 수 있게 하는 연구도 유망하다. 일론 머스크가 진행하는 뉴럴 링크 사업도 그
과정의 일환으로 볼 수 있겠다.