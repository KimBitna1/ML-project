## State of Charge (SOC) estimation for electric Vertical Take-Off and Landing aircraft (eVTOL) batteries

### Introduction

  리튬 이온 배터리는 전기차, 드론, 항공기 등 다양한 시스템에서 사용되는 중요한 에너지 저장 장치이다. 배터리를 안전하고 효율적으로 사용하기 위해서는 현재 배터리에 얼마나 많은 에너지가 남아있는지 정확하게 알아야 한다. 

  이때 사용되는 대표적인 지표는 SOC (State of Charge)이다. SOC란 특정 온도와 discharge rate (한 시간 동안 사용할 수 있는 전류량)등의 조건에서 배터리가 손상되지 않고 제공할 수 있는 최대 방전 용량을 의미한다. 이는 배터리의 nominal capacity (최대 가용 전하량)에 대해 남아있는 capacity의 비로 나타낸다.

$$ 
\mathrm{SOC} = \frac{Q_\text{remaining}}{Q_\text{nominal}}
$$


  SOC를 정확하게 아는 것은 운행 가능 시간 예측, 시스템 제어, 안전 관리 등을 위해 중요하다. 
그러나 실제 배터리 시스템에서 SOC를 직접적으로 측정할 수 없다. 배터리에서 실제로 측정 가능한 것은 전압, 전류, 온도이며, 이를 통해 우리는 SOC를 예측해야 한다. 
남아있는 capacity의 양은 충전 혹은 방전 동안 흐르는 전류를 적분함으로써 구할 수 있다. 따라서 SOC는 다음과 같은 식으로도 표현된다.

$$
\mathrm{SOC}(t) = \mathrm{SOC}_0 - \frac{\eta \int_{t_0}^{t} i(t)\ dt}{Q_n}
$$

여기서 $\mathrm{SOC}_0$ 은 initial SOC를, $\eta$는 efficiency factor를 나타낸다. 실제로는 배터리가 노화됨에 따라 $Q_n$ (i.e., 최대 가용 전하량)이 감소하고, charge/discharge rate, 온도, self-discharge, aging 등을 고려해서 보정해야 한다.
