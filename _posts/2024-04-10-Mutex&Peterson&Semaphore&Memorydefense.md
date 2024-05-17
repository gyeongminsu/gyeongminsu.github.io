---
title : Multithreading의 Race condition을 해소하기 위한 방법(Peterson's algorithm, MemoryDefense, Semaphore)
categories : Operating_System, Multithreading, Peterson's_algorithm
tags : Operating_System, Multithreading, Peterson's_algorithm
date : 2024-04-09 12:00:00 +0900
pin : true
path : true
math : true
toc : true
layout : post
comments : true
---



# 1. 기본적인 상황

![과제1-1.png](/assets/img/2024-04-10-Mutex&Peterson&Semaphore&Memorydefense/%25EA%25B3%25BC%25EC%25A0%259C1-1.png)

처음에 주어진 Program.cs 코드를 실행하면 Main 함수에서 선언된 Thread t가 ThreadBody()에서 shared_var에 접근하고, Main 함수 스레드가 shared_var에 접근하여 총 두 개의 스레드가 공유 변수(volatile static int)인 shared_var에 접근하는 것을 알 수 있다.

초기 상태에서 아무런 제어 없이 공유 자원에 두 개의 스레드가 접근하면 Race Condition이 발생하여 짜여진 코드대로 공유 자원에 올바르게 접근하지 못 하는 결과를 부르게 된다.

따라서 코드 상에서 shared_var에 접근이 올바르게 이루어진다면, 출력되는 shared_var의 값은 200,000,000이어야 하지만 실제로는 Race Condition 때문에 그보다 적은 값으로 출력되는 것을 볼 수 있다.

- “Race Condition”이란?
    - 여러(두 개 이상) 프로세스 혹은 여러 스레드가 똑같은 공유 자원에 접근하려 할 때, 서로 공유 자원을 사용하기 위해 경합(Race)을 벌이는 현상을 말한다.
    - 한 프로세스 내에서 여러 스레드를 사용하는 멀티 스레딩(Multi Threading)을 진행할 때, 각 스레드는 프로세스 내에 할당된 메모리인 Heap 영역과 Data 영역을 공유하여 사용한다.
    - 이 때 스레드 간의 실행 순서를 올바르게 지정하지 않으면, 프로세스를 실행할 때마다 스레드 간의 실행 순서가 뒤바뀌어 공유 자원 접근이 올바르게 이루어지지 않고 매번 프로세스의 실행 결과가 달라질 수 있다.

# 2. 동기화 기법 중 Peterson’s Algorithm을 이용하여 shared_var을 증가시킬 때 상호 배제가 되도록 수정하여 실행하라. 프로그램과 결과(화면 캡쳐)를 보고서에 첨부하라.

Peterson’s Algorithm(이하 피터슨 알고리즘이라 부른다.)은 수학자 개리 피터슨(Gary Peterson)이 1981년에 발표한 알고리즘으로, 공유 메모리를 활용하여 여러 프로세스 또는 여러 스레드가 하나의 자원을 사용할 때 **상호 배제(Mutex)**를 적용하여 문제(Race Condition)가 발생하지 않도록 해 준다. 

상호 배제를 알아보기 전에 **임계 구역(Critical Region)**에 대해 먼저 알아보도록 한다.

임계 구역이란 멀티스레딩을 진행할 때 둘 이상의 스레드가 공유 자원에 동시에 접근하지 못 하도록 코드 상에서 시간을 지정하여 한 번에 하나의 스레드만 공유 자원에 접근하도록 만든 개념이다. 어떤 스레드가 이미 임계 구역에 들어가서 공유 자원에 접근 중이라면, 다른 스레드는 지정된 시간만큼 대기한 후 임계 구역에 들어가서 공유 자원에 접근할 수 있다.

임계 구역을 통해 Race Condition을 해결하기 위해 만족해야 하는 세 가지의 조건(Condition)이 존재한다.

- 상호 배제(Mutex)
    - 임계 구역에는 시간적으로 동시에 여러 스레드가 진입할 수 없다.
- 진행(Progress)
    - 임계 구역 밖에서 실행되고 있는 스레드가 다른 스레드의 실행을 막을 수 없다.
- 한정된 대기(Bounded Waiting)
    - 어떤 스레드도 임계 구역에 들어가기 위해 영원히 대기하여선 안 된다.

피터슨 알고리즘에서는 Mutex를 구현하기 위해 Peterson Class에 boolean 변수인 flag, 그리고 int 변수인 turn을 사용한다.

아래의 코드에서는 스레드가 두 개만 실행된다고 가정하였으므로 flag가 선언된 배열의 크기를 2로 하고, threadId의 unique 개수를 2로 하였다.

그리고 임계 구역에 진입하는 메소드인 EnterRegion을 threadId를 파라미터로 하여 만들어주었다. 앞서 언급하였듯이 threadId의 unique는 0, 1밖에 없으므로 다른 스레드의 Id인 otherThreadId 변수는 1 - threadId로 선언하였다.

그리고 현재 실행되는 thread의 flag를 true로 바꿔주고, turn은 otherThreadId로 바꿔주어 임계 구역 진입 조건을 만족시켜주었다.

```csharp
internal class Program
{
    volatile static int shared_var;
    static void ThreadBody(object threadIdObj)
    {
        int threadId = (int)threadIdObj;
        for (int i = 0; i < 10000; i++)
        {
            for (int j = 0; j < 10000; j++)
            {
                Peterson.EnterRegion(threadId);
                shared_var++;
                Peterson.LeaveRegion(threadId);
            }
        }
    }

    public static class Peterson
    {
        public static bool[] flag = new bool[2];
        public static int turn;

        public static void EnterRegion(int threadId)
        {
            int otherThreadId = 1 - threadId;
            flag[threadId] = true;
            turn = otherThreadId;

            while (flag[otherThreadId] && turn == otherThreadId)
            {
                // Busy waiting
            }
        }

        public static void LeaveRegion(int threadId)
        {
            flag[threadId] = false;
        }
    }

    private static void Main(string[] args)
    {
        shared_var = 0;

        Thread t = new Thread(() => ThreadBody(0));
        t.Start();

        ThreadBody(1);

        t.Join();

        Console.WriteLine(shared_var);
        Console.WriteLine("2024 Spring 운영체제 수업 2020920001 경민수 컴퓨터과학부");
    }
}
```

![Untitled](/assets/img/2024-04-10-Mutex&Peterson&Semaphore&Memorydefense/Untitled.png)

$\uparrow$ Peterson’s algorithm을 이용한 shared_var 코드 실행 결과 사진

피터슨 알고리즘을 이용하여 Mutex를 구현한 후 두 개의 스레드로 공유 자원에 접근하여 shared_var를 증가시켰다. 

과제 1의 코드보다 훨씬 더 많은 접근이 이루어져 shared_var의 값이 200,000,000에 가까워지긴 했지만 여전히 37,768번의 접근이 이루어지지 않은 것을 볼 수 있다.

여기서 우리는 shared_var의 증가 연산이 원자적(atomic)으로 진행되었는지 확인할 필요가 있다.

왜냐하면 피터슨 알고리즘은 스레드 간의 실행이 원자적이어야 Race condition의 해결이 보장되기 때문이다.

shared_var는 **‘volatile’** 키워드로 선언되었다. C/C++에서 ‘volatile’ 키워드로 선언된 변수는 컴파일러가 해당 변수를 최적화하는 것을 방지하고 항상 메모리에서 바로 읽혀도록 보장해준다.

그치만 volatile의 이런 특성이 공유 변수의 원자적 실행을 보장하진 않는다. 다른 방법을 추가하여 원자적 실행을 구현할 필요가 있다.

# 3. mfence와 같은 메모리 배리어 명령이 무엇지 조사한다. Peterson’s Algorithm 코드의 적절한 부분에 Thread.MemoryBarrier();를 추가한다. 그리고 실험 결과를 레포트에 제시하라. 프로그램과 결과(화면 캡쳐)를 보고서에 첨부하라.

메모리 배리어 명령은 컴파일러나 CPU에게 특정 연산 순서를 메모리 배리어 명령어 전후로 강제하도록 하는 명령이다.

멀티스레딩을 하는 경우 컴파일러가 자체 최적화를 실행하여 스레드 간의 실행 순서가 뒤바뀌는 것을 방지하기 위해 특정 부분(이 보고서에서는 임계 구역)에서 스레드의 실행 순서를 강제로 지정하여 스레드 간의 간섭을 방지할 필요가 있다.

이러한 기능을 메모리 배리어 명령을 통해 실현할 수 있다.

아래 코드에서는 두 개의 위치에 Thread.MemoryBarrier() 명령어를 배치하였다.

1.  Thread가 공유 변수인 shared_var에 접근하여 값을 변경하는 shared_var++ 전후에 MemoryBarrier를 배치하여 스레드 간의 실행 순서가 변경되지 않도록 하였다.
2. 임계 구역에 진입하는 메소드인 Peterson.EnterRegion과 임계 구역에서 벗어나는 메소드인 Peterson.LeaveRegion에서 flag와 turn 변수의 값을 설정하고 검사하기 전후에 MemoryBarrier를 배치하여 다른 스레드가 변수 변경에 접근하지 못 하도록 제어한다.

```csharp
internal class Program
{
    volatile static int shared_var;
    static void ThreadBody(object threadIdObj)
    {
        int threadId = (int)threadIdObj;
        for (int i = 0; i < 10000; i++)
        {
            for (int j = 0; j < 10000; j++)
            {
                Peterson.EnterRegion(threadId);
                
                // 메모리 배리어를 사용하여 연산 전의 메모리 작업 순서 조정
                Thread.MemoryBarrier();
                shared_var++;
                // 메모리 배리어를 사용하여 연산 후의 메모리 작업 순서 조정
                Thread.MemoryBarrier();
                
                Peterson.LeaveRegion(threadId);
            }
        }
    }

    public static class Peterson
    {
        public static bool[] flag = new bool[2];
        public static int turn;

        public static void EnterRegion(int threadId)
        {
            int otherThreadId = 1 - threadId;
            flag[threadId] = true;
            
            Thread.MemoryBarrier();
            turn = otherThreadId;
						
						Thread.MemoryBarrier();
            while (flag[otherThreadId] && turn == otherThreadId)
            {
                // Busy waiting
            }
        }

        public static void LeaveRegion(int threadId)
        {
            flag[threadId] = false;
        }
    }

    private static void Main(string[] args)
    {
        shared_var = 0;

        Thread t = new Thread(() => ThreadBody(0));
        t.Start();

        ThreadBody(1);

        t.Join();

        Console.WriteLine(shared_var);
        Console.WriteLine("2024 Spring 운영체제 수업 2020920001 경민수 컴퓨터과학부");
    }
}
```

![Untitled](/assets/img/2024-04-10-Mutex&Peterson&Semaphore&Memorydefense/Untitled%201.png)

$\uparrow$ Thread.MemoryBarrier() 명령어를 통한 스레드의 원자적 실행 구현

메모리 배리어를 2번 과제인 피터슨 알고리즘 코드의 적절한 위치에 추가하여 스레드의 원자적 실행을 구현하였다. 덕분에 shared_var의 값이 정상적으로 200,000,000으로 출력되는 것을 볼 수 있다.

# 4. Windows는 Mutex, Event, Semaphore와 같은 동기화 기법을 제공한다. 이들에 대해 조사하여 기능과 사용법을 레포트에 기술하라.

윈도우 운영체제에서는 공유 자원에 접근할 때 **동기화 개체**를 이용하여 동기화를 진행한다.

스레드는 대기 조건이 충족되면 반환하는 대기 함수를 호출하여 실행을 멈춘 후, 동기화 개체를 선언한다.

아래의 내용에는 Mutex, Event, Semaphore에서 윈도우 운영체제가 동기화 개체를 이용하여 동기화를 진행하는 방법을 설명한다.

## 4-1. Mutex

윈도우 운영체제에서는 스레드가 뮤텍스 개체를 소유함으로써 뮤텍스 동기화를 시행한다.

코드 상에서 스레드가 공유 자원에 접근할 때, 뮤텍스 개체를 통해 뮤텍스의 소유권을 가져야 접근할 수 있다. 

여러 스레드가 동시에 공유 자원에 접근하려는 경우, 뮤택스 개체를 가진 하나의 스레드만 공유 자원에 접근 가능하다. 

스레드의 공유 자원 접근이 끝난 후, 스레드는 뮤택스 개체를 해제함으로써 다른 스레드가 뮤택스 개체를 소유할 수 있게 한다.

- 우선 스레드는 ‘**CreateMutex**’ 함수를 호출하여 뮤텍스 개체를 만들고, 뮤텍스 개체의 소유권을 요청할 수 있다. 또한 뮤텍스 개체의 이름도 지정할 수 있다.
- 이 때 다른 스레드는 ‘**OpenMutex**’ 함수에서 기존에 선언되어 명명된 뮤텍스 개체를 호출하여 뮤텍스 개체에 대한 핸들링을 진행할 수 있다. 핸들링을 진행하는 스레드는 대기 함수를 호출하여 뮤텍스 개체의 소유권을 요청할 수 있다.
- 이렇게 다른 스레드가 소유권을 요청하는 경우 기존에 뮤텍스 개체를 소유하고 있던 스레드는 ‘**ReleaseMutex**’ 함수를 호출하여 핸들링 스레드가 뮤텍스 개체를 해제할 때 까지 다른 스레드가 뮤텍스 개체를 요청하는 것을 차단한다.

## 4-2. Event

멀티 스레딩을 할 때 Event는 스레드 간의 상태를 알리고 동기화를 조절할 때 사용한다. Event의 유형에는 이벤트 발생 후 수동으로 리셋해야 하는 Manual-Reset Event와 이벤트 발생 후 자동으로 리셋되는 **Auto-Reset Event**로 나뉜다.

스레드는 여러 상황에서 이벤트 개체를 사용하여 대기 스레드에 이벤트를 알릴 수 있다. 

- 스레드는 ‘**CreateEvent**’ 함수를 호출하여 초기 상태가 설정되지 않은 이벤트 개체를 생성하고, 이벤트 개체의 이름도 지정할 수 있다.
- ‘**SetEvent**’ 함수를 호출하여 이벤트의 상태를 시그널할 수 있다. 그리고 Auto-Reset Event의 경우 이벤트 개체가 자동으로 리셋되지만, Manual-Reset Event의 경우  ‘**ResetEvent**’ 함수를 호출하여 이벤트를 비시그널된 상태로 리셋한다.
- **‘WaitForSingleObeject’** 함수를 호출하여 이벤트를 대기시킬 수 있다. 이 함수는 이벤트가 시그널될 때 까지 스레드를 블로킹한다. 이벤트가 시그널되면 스레드가 다시 실행하고 이벤트를 처리할 수 있다.

## 4-3. Semaphore

세마포어는 멀티 스레딩을 통한 공유 자원 접근을 할 때, 특정 개수 이하의 스레드만 공유 자원에 접근할 수 있도록 공유 자원을 보호하는 데 사용된다.

세마포어 개체는 카운터 값을 가지고 있고, 스레드는 세마포어의 카운터 값을 증가시키거나 감소시켜 조절할 수 있다. 세마포어의 카운터 값이 0보다 클 경우 스레드는 세마포어를 소유할 수 있고, 카운터 값이 0이면 세마포어를 소유할 수 없다. 이 경우 스레드는 카운터 값이 0에서 증가할 때 까지 대기하여야 한다.

- 스레드는 ‘**CreateSemaphore**’ 함수를 호출하여 세마포어 개체를 만든다. 이 때 세마포어 개체의 초기 개수와 세마포어 개체 수의 최댓값을 지정한다. 초기 개수의 범위는 0에서 개체 수의 최댓값 사이여야 한다. 또한 세머포어 개체의 이름을 지정할 수 있다.
- 스레드가 공유 자원에 접근하기 전에 ‘**WaitForSingleObject**’ 함수를 호출하여 세마포어 개체의 카운터가 접근을 허용하는 지, 즉 카운터 값이 0보다 큰 지 확인한다. 이 때 카운터의 값이 1보다 크면 접근을 허용하고, WaitForSingleObject 함수는 세마포어 카운터의 수를 1씩 감소시킨다.
- 스레드가 공유 자원에 접근하는 작업을 완료하면 ‘**ReleaseSemaphore**’ 함수를 호출하여 다시 세마포어 카운터의 수를 1씩 증가시킨다. 이를 통해 다른 대기 스레드가 공유 자원에 대한 접근을 수행할 수 있다.