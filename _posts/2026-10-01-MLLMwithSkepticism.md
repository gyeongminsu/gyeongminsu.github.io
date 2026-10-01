---
title : MLLMs for Philosophy — 회의주의로 읽는 멀티모달 모델
categories : [MLLM, Skepticism, Epistemology, Moore, ComputerVision]
tags : [MLLM, Skepticism, Epistemology, Moore, ComputerVision]
date : 2026-10-01 18:00:00 +0900
pin : true
path : true
math : true
image : /assets/img/2026-10-01-MLLMwithSkepticism/thumbnail.jpg
toc : true
layout : post
comments : true
---

# 0. Introduction

작년 11월, 학과 소모임에서 [언어 모델과 그를 둘러싼 질문들]({% post_url 2025-11-21-LLMwithquestion %})이라는 제목으로 발표를 했다. 그 발표가 언어 모델을 중심으로 언어학, 과학철학, 대륙철학을 맛보는 자리였다면, 이번에는 그 후속으로 멀티모달 모델의 인식론을 다뤄 보았다.

이번 발표는 2026년 가을학기 컴퓨터비전 대학원 수업에서 진행했다. 청중은 인공지능을 연구하는 대학원생과 교수님들이었고, 대부분 철학에는 익숙하지 않았다. 그래서 철학 개념이 나올 때마다 ML 용어로 바꿔 말하는 방식을 택했다.

처음에는 접지, 지각과 판단, 지각의 오프로딩, 지적 자율성까지 네 갈래를 개관하는 구성도 만들어 보았다. 최종적으로는 회의주의라는 논증 하나를 렌즈로 삼아 시각언어 논문 세 편을 깊게 읽는 구성으로 정리했다. 슈토이프의 『현대 인식론 입문』 회의주의 장에서 출발해 무어의 「외부 세계의 증명」을 거쳐, 멀티모달 모델이 그 논증의 어디에 서 있는지 묻는 흐름이다.

이제 발표 슬라이드와 슬라이드 내용에 관한 간단한 첨언들을 시작하겠다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0001.jpg)

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0002.jpg)

# 1. Why Skepticism?

회의주의가 던지는 질문은 세 가지로 정리할 수 있다. 내 지각을 믿을 수 있는가, 정상적인 경우와 속은 경우를 구별할 수 있는가, 남의 눈을 믿을 수 있는가. 흥미로운 점은 이 질문들이 ML에서 이미 환각, 딥페이크, 과의존이라는 이름으로 다뤄지고 있다는 것이다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0004.jpg)

통 속의 뇌는 신경 말단으로 들어오는 신호만 받을 뿐, 그 신호를 만든 컴퓨터는 볼 수 없다. 멀티모달 모델도 마찬가지로, 입력된 픽셀이 카메라에서 왔는지 생성기나 편집기에서 왔는지 알 수 없다. 그래서 이미지 업로드는 통의 비유가 아니라 통을 문자 그대로 묘사한 것이라고 보았다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0005.jpg)

회의주의 논증은 네 단계로 정리된다. 전달성 원칙과 "두 손이 있다면 통 속의 뇌가 아니다"라는 논리적 사실은 포기하기 어렵다. 그래서 회의주의에 대한 모든 대답은 결국 (3), 즉 결정적 증거 없이 "나는 나쁜 경우에 있지 않다"를 정당화할 수 있느냐에 대한 대답이 된다. 3부의 논문들은 모델이 바로 이 (3)에서 무엇을 하는지 보여 준다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0006.jpg)

# 2. "Here is one hand."

무어는 회의주의자와 같은 조건문을 받아들이되, 반대 방향으로 추론한다. 회의주의자는 전건 긍정으로 "두 손이 있다는 믿음도 정당화되지 않는다"고 결론 내리고, 무어는 후건 부정으로 "통 속의 뇌가 아니라는 믿음이 정당화된다"고 결론 내린다. 두 추론 모두 타당하므로, 싸움은 논리가 아니라 어느 전제가 더 무거운가에 있다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0008.jpg)

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0009.jpg)

1939년 영국 학술원 강연에서 무어는 실제로 두 손을 들어 보이며 "여기 한 손이 있다"고 말했다. 인식론의 표준적인 판정은, 회의주의가 확실성은 무너뜨리지만 일상적 지식은 무너뜨리지 못한다는 것이다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0010.jpg)

그런데 오류 가능한 정당화가 일상에서 충분한 것은 정상 조건 아래서 작동하기 때문이다. 모델이 in-distribution에서만 믿을 만한 것과 같은 구조다. 이 발표에서는 정상 조건을 세 가지로 나누었다. 가리키는 것과 보는 것이 하나의 행위라는 것, 지각이 이론보다 무겁다는 것, 그리고 가짜 손이 드물다는 것이다. 사람에게 이 셋은 확인할 필요조차 없는 배경이지만, 모델에게는 각각이 측정 가능한 변수가 된다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0011.jpg)

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0012.jpg)

# 3. What's shown in MLLMs

## (a) Liu et al. — 가리키지만 보지 않는다

Liu et al.은 멀티모달 추론 모델이 답하기 전에 이미지를 확대해서 다시 보는 "visual thought"에 직접 개입했다. 텍스트를 건드리면 정확도가 크게 떨어지지만, 확대 이미지를 노이즈로 바꿔도 답은 거의 변하지 않았다. 노이즈를 보고도 모델은 "드레스가 보인다"고 쓴다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0014.jpg)

정답을 맞혔지만 근거로 삼은 확대 영역은 엉뚱하거나 부족한 사례도 있다. 정확도는 답만 보기 때문에 이것을 성공으로 집계하지만, 회의주의자는 증거가 답을 뒷받침하는지를 묻는다. 그 기준에서 조건 (a)는 모델에게 성립하지 않는다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0015.jpg)

## (b) Gavrikov et al. — 문장이 판정을 움직인다

Gavrikov et al.은 형태와 질감이 충돌하는 이미지로 VLM의 shape bias를 재고, 프롬프트만으로 그 편향을 49%에서 72%까지 움직였다. vision encoder는 두 단서를 모두 갖고 있고, 그중 하나를 고르는 것은 LLM이다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0016.jpg)

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0017.jpg)

이 결과는 회의적 가설의 둘째 특징, 즉 증거와 완벽하게 양립한다는 성질이 문자 그대로 구현된 경우로 읽을 수 있다. 이미지가 두 답을 모두 지지하니 판정은 이미지 바깥에서 와야 하고, 그것이 LLM과 프롬프트다. 사람도 지시를 받으면 형태 편향이 떨어지므로 조종 가능성 자체가 문제는 아니다. 문제는 누가 조종하는지, 그리고 그 지시가 사용자에게 보이는지다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0018.jpg)

## (c) Bowers et al. — 알아볼 수 없는 가짜, 구별하지 못하는 점수

Bowers et al.은 벤치마크에서 최고 점수를 받는 DNN이 통제된 심리학 실험의 결과는 대부분 재현하지 못한다는 점을 검토한 리뷰 논문이다. 여기서 다룬 사례들은 MLLM이 아니라 이미지 분류기에서 나온 결과라는 점은 밝혀 둔다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0019.jpg)

이 논문을 회의주의적으로 읽으면 두 가지 문제가 나온다. 모델의 문제는, 골드먼의 가짜 헛간과 달리 모델을 속이는 가짜가 우리 눈에는 잡음이나 픽셀 하나로 보여서 무엇을 배제해야 할지 목록을 쓸 수 없다는 것이다. 우리의 문제는, 지름길을 쓰는 네트워크도 RSA 점수가 높게 나올 수 있어서 점수만으로는 보는 모델과 지름길 모델을 구별할 수 없다는 것이다. 둘을 구별하는 방법은 증거를 바꾸고 답이 따라오는지 보는 개입이다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0020.jpg)

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0021.jpg)

# 4. Nevertheless

그런데도 사람들은 이미 이 모델들을 통해 세상을 보고 있다. 시각장애·저시력 사용자의 일기 연구에서 후속 답의 22%에 환각이 있었지만, 신뢰도는 높게 유지되었다. 무어는 자기 손을 직접 봤지만, 시각장애인 사용자는 "여기 한 손이 있다"를 모델의 증언으로 받는다. 모델의 실패는 모델 안에 머물지 않고 사용자에게 넘어간다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0023.jpg)

한편 사람에게 회의주의는 끝내 논증으로 남지만, 모델은 밖에서 들여다볼 수 있다는 점이 다르다. 마지막 장에서는 이 논문들이 가능하게 하는 세 가지로 무어 지수, 규범적 결정, 은닉 상태 프로브를 제안했다. 정상 조건을 측정하고, 실패한 조건을 고치고, 조건이 실패할 때 사용자에게 보여 주는 것이 남은 일이다.

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0024.jpg)

![image.jpg](/assets/img/2026-10-01-MLLMwithSkepticism/jpg-0025.jpg)
