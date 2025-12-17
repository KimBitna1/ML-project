## State of Charge (SOC) estimation for electric Vertical Take-Off and Landing aircraft (eVTOL) batteries

### Introduction

 리튬 이온 배터리는 전기차, 드론, 항공기 등 다양한 시스템에서 사용되는 중요한 에너지 저장 장치이다. 배터리를 안전하고 효율적으로 사용하기 위해서는 현재 배터리에 얼마나 많은 에너지가 남아있는지 정확하게 알아야 한다. 

이때 사용되는 대표적인 지표는 SOC (State of Charge)이다. SOC란 특정 온도와 discharge rate 등의 조건에서 배터리가 손상되지 않고 제공할 수 있는 최대 방전 용량을 의미한다. 이는 배터리의 rated capacity (최대 가용 전하량)에 대해 남아있는 capacity의 비로 나타낸다.
$$ 
SOC(t) = Q(t)/Q(max)
$$

