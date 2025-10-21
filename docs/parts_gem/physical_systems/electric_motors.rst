Electric Motors
###############

Electric Motor Base Class
*************************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.ElectricMotor
   :members:


Synchronous Motors
******************

Parameter Dictionary
''''''''''''''''''''

+------------------+------------------------------------------------+---------------------+
| **Key**          |  **Description**                               | **Default**         |
+==================+================================================+=====================+
| r_s              | Stator Resistance in Ohm                       | 4.9                 |
+------------------+------------------------------------------------+---------------------+
| l_d              | d-axis inductance in Henry                     | 79e-3               |
+------------------+------------------------------------------------+---------------------+
| l_q              | q-axis inductance in Henry                     | 113e-3              |
+------------------+------------------------------------------------+---------------------+
| j_rotor          | Moment of inertia of the rotor                 | 2.45e-3             |
+------------------+------------------------------------------------+---------------------+
| psi_p            | Permanent linked rotor flux                    | 0.165               |
+------------------+------------------------------------------------+---------------------+
| p                | Pole pair Number                               | 2                   |
+------------------+------------------------------------------------+---------------------+


All nominal voltages and currents are peak phase values.
Therefore, data sheet values for line voltages and phase currents has to be transformed such that
:math:`U_N=\sqrt(2/3) U_L` and :math:`I_N=\sqrt(2) I_S`.

Furthermore, the angular velocity is the electrical one and not the mechanical one :math:`\omega = p \omega_{me}`.


.. autoclass:: gym_electric_motor.physical_systems.electric_motors.SynchronousMotor
   :members:

Synchronous Reluctance Motor
****************************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.SynchronousReluctanceMotor
   :members:

Permanent Magnet Synchronous Motor
**********************************

The PMSM is a three phase motor with a permanent magnet in the rotor as shown in the figure [Boecker2018b]_.
The input of this motor are the voltages :math:`u_a`, :math:`u_b` and :math:`u_c`.

The quantities are:

- :math:`u_a`, :math:`u_b`, :math:`u_c` phase voltages

- :math:`i_a`, :math:`i_b`, :math:`i_c` phase currents

- :math:`R_s` stator resistance

- :math:`L_d` d-axis inductance

- :math:`L_q` q-axis inductance


- :math:`i_{sd}` d-axis current

- :math:`i_{sq}` q-axis current

- :math:`u_{sd}` d-axis voltage

- :math:`u_{sq}` q-axis voltage

- :math:`p` pole pair number

- :math:`\mathit{\Psi}_p` permanent linked rotor flux

- :math:`\epsilon` rotor position angle

- :math:`\omega` (electrical) angular velocity

- :math:`\omega_{me}` mechanical angular velocity

- :math:`T` Torque produced by the motor

- :math:`T_L` Torque from the load

- :math:`J` moment of inertia

The electrical angular velocity and the mechanical angular velocity are related such that :math:`\omega=\omega_{me} p`.

.. figure:: ../../plots/GDAFig29.svg

The circuit diagram of the phases are similar to each other and the armature circuit of the externally excited motor.

.. figure:: ../../plots/pmsmMotorB6.png

For an easy computation the three phases are first transformed to the quantities :math:`\alpha` and :math:`\beta` and
afterwards to :math:`d/q` coordinates that rotated with the rotor as given in [Boecker2018b]_.

.. figure:: ../../plots/ESBdq.svg

This results in the equations:


:math:`u_{sd}=R_s i_{sd}+L_d \frac{\mathrm{d} i_{sd}}{\mathrm{d} t}-\omega_{me}p L_q i_{sq}`

:math:`u_{sq}=R_s i_{sq}+L_q \frac{\mathrm{d} i_{sq}}{\mathrm{d} t}+\omega_{me}p L_d i_{sd}+\omega_{me}p \mathit{\Psi}_p`

:math:`\frac{\mathrm{d} \omega_{me}}{\mathrm{d} t}=\frac{T-T_L(\omega_{me})}{J}`

:math:`T=\frac{3}{2} p (\mathit{\Psi}_p +(L_d-L_q)i_{sd}) i_{sq}`



A more detailed derivation can be found in
[Modeling and High-Performance Control of Electric Machines, John Chiasson (2005)]

The difference between rms and peak values and between line and phase quantities has to be considered at the PMSM.
The PMSM is in star conncetion and the line voltage :math:`U_L` is mostly given in data sheets as rms value.
In the toolbox the nominal value of the phase voltage :math:`\hat{U}_S=\sqrt{\frac{2}{3}}U_L` is needed.
Furthermore, the supply voltage is typically the same :math:`u_{sup}=\hat{U}_S`.
For example, a line voltage of :math:`U_L=400~\text{V}` is given, the rms phase voltage is
:math:`U_S=\sqrt{\frac{1}{3}}U_L = 230.9 \text{ V}`
and the peak value :math:`\hat{U}_S=326.6 \text{ V}`.
The nominal peak current of a phase is given by :math:`\hat{I}_S=\sqrt{2} I_S`.

.. figure:: ../../plots/Drehstromtrafo.svg


.. autoclass:: gym_electric_motor.physical_systems.electric_motors.PermanentMagnetSynchronousMotor
   :members:



References
##########

.. [Boecker2018a] Böcker, Joachim; Elektrische Antriebstechnik; 2018; Paderborn University

.. [Boecker2018b] Böcker, Joachim; Controlled Three-Phase Drives; 2018; Paderborn University

.. [Chiasson2005] Chiasson, John; Modeling and High-Performance Control of Electric Machines; 2005; Hoboken, NJ, USA



