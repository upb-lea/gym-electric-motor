from gym.envs.registration import register

envs_path = 'gym_electric_motor.envs:'

register(id='emotor-dc-extex-cont-v0',
         entry_point=envs_path+'DCMExtExCont')
register(id='emotor-dc-extex-disc-v0',
         entry_point=envs_path+'DCMExtExDisc')

register(id='emotor-dc-permex-cont-v0',
         entry_point=envs_path+'DCMPermExCont')
register(id='emotor-dc-permex-disc-v0',
         entry_point=envs_path+'DCMPermExDisc')

register(id='emotor-dc-series-cont-v0',
         entry_point=envs_path+'DCMSeriesCont')
register(id='emotor-dc-series-disc-v0',
         entry_point=envs_path+'DCMSeriesDisc')

register(id='emotor-dc-shunt-cont-v0',
         entry_point=envs_path+'DCMShuntCont')
register(id='emotor-dc-shunt-disc-v0',
         entry_point=envs_path+'DCMShuntDisc')

"""register(id='emotor-asm-cont-v0',
         entry_point=envs_path+'ASMContinuousControlEnv')
register(id='emotor-asm-disc-v0',
         entry_point=envs_path+'ASMContinuousControlEnv')"""

register(id='emotor-pmsm-cont-v0',
         entry_point=envs_path+'PmsmCont')
register(id='emotor-pmsm-disc-v0',
         entry_point=envs_path+'PmsmDisc')
