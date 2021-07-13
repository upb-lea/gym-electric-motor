Search.setIndex({docnames:["JOSS/paper","index","parts/callbacks","parts/constraint_monitor","parts/constraints/constraint","parts/constraints/limit_constraint","parts/constraints/squared_constraint","parts/core","parts/environments/dc_extex_cont","parts/environments/dc_extex_disc","parts/environments/dc_permex_cont","parts/environments/dc_permex_disc","parts/environments/dc_series_cont","parts/environments/dc_series_disc","parts/environments/dc_shunt_cont","parts/environments/dc_shunt_disc","parts/environments/dfim_cont","parts/environments/dfim_disc","parts/environments/environment","parts/environments/pmsm_cont","parts/environments/pmsm_disc","parts/environments/scim_cont","parts/environments/scim_disc","parts/environments/synrm_cont","parts/environments/synrm_disc","parts/physical_systems/converters/1QC","parts/physical_systems/converters/2QC","parts/physical_systems/converters/4QC","parts/physical_systems/converters/B6C","parts/physical_systems/converters/DoubleConv","parts/physical_systems/converters/NoConv","parts/physical_systems/converters/converter","parts/physical_systems/electric_motors","parts/physical_systems/electric_motors/dc_base","parts/physical_systems/electric_motors/dfim","parts/physical_systems/electric_motors/electric_motor","parts/physical_systems/electric_motors/extex","parts/physical_systems/electric_motors/induction_base","parts/physical_systems/electric_motors/permex","parts/physical_systems/electric_motors/pmsm","parts/physical_systems/electric_motors/scim","parts/physical_systems/electric_motors/series","parts/physical_systems/electric_motors/shunt","parts/physical_systems/electric_motors/synchronous_base","parts/physical_systems/electric_motors/synrm","parts/physical_systems/electric_motors/three_phase_base","parts/physical_systems/mechanical_loads/const_speed_load","parts/physical_systems/mechanical_loads/ext_speed_load","parts/physical_systems/mechanical_loads/mechanical_load","parts/physical_systems/mechanical_loads/polystatic","parts/physical_systems/noise_generators/gaussian_white_noise","parts/physical_systems/noise_generators/noise_generator","parts/physical_systems/ode_solvers/euler","parts/physical_systems/ode_solvers/ode_solver","parts/physical_systems/ode_solvers/scipy_ode","parts/physical_systems/ode_solvers/scipy_odeint","parts/physical_systems/ode_solvers/scipy_solve_ivp","parts/physical_systems/physical_system","parts/physical_systems/scml_system","parts/physical_systems/voltage_supplies/ac_1_phase_supply","parts/physical_systems/voltage_supplies/ac_3_phase_supply","parts/physical_systems/voltage_supplies/ideal_voltage_supply","parts/physical_systems/voltage_supplies/rc_voltage_supply","parts/physical_systems/voltage_supplies/voltage_supply","parts/readme","parts/reference_generators/const_reference_generator","parts/reference_generators/multiple_ref_generator","parts/reference_generators/reference_generator","parts/reference_generators/sawtooth_reference_generator","parts/reference_generators/sinusoidal_reference_generator","parts/reference_generators/step_reference_generator","parts/reference_generators/subepisoded_reference_generator","parts/reference_generators/switched_reference_generator","parts/reference_generators/triangular_reference_generator","parts/reference_generators/wiener_process_reference_generator","parts/reference_generators/zero_reference_generator","parts/reward_functions/reward_function","parts/reward_functions/weighted_sum_of_errors","parts/technical_models","parts/technicalbackground","parts/utils","parts/visualizations/console_printer","parts/visualizations/motor_dashboard","parts/visualizations/motor_dashboard_plots/action_plot","parts/visualizations/motor_dashboard_plots/cumulative_constraint_violation_plot","parts/visualizations/motor_dashboard_plots/episode_length_plot","parts/visualizations/motor_dashboard_plots/episode_plots","parts/visualizations/motor_dashboard_plots/mean_episode_reward_plot","parts/visualizations/motor_dashboard_plots/reward_plot","parts/visualizations/motor_dashboard_plots/state_plot","parts/visualizations/motor_dashboard_plots/step_plot","parts/visualizations/motor_dashboard_plots/time_plot","parts/visualizations/visualization"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["JOSS/paper.md","index.rst","parts/callbacks.rst","parts/constraint_monitor.rst","parts/constraints/constraint.rst","parts/constraints/limit_constraint.rst","parts/constraints/squared_constraint.rst","parts/core.rst","parts/environments/dc_extex_cont.rst","parts/environments/dc_extex_disc.rst","parts/environments/dc_permex_cont.rst","parts/environments/dc_permex_disc.rst","parts/environments/dc_series_cont.rst","parts/environments/dc_series_disc.rst","parts/environments/dc_shunt_cont.rst","parts/environments/dc_shunt_disc.rst","parts/environments/dfim_cont.rst","parts/environments/dfim_disc.rst","parts/environments/environment.rst","parts/environments/pmsm_cont.rst","parts/environments/pmsm_disc.rst","parts/environments/scim_cont.rst","parts/environments/scim_disc.rst","parts/environments/synrm_cont.rst","parts/environments/synrm_disc.rst","parts/physical_systems/converters/1QC.rst","parts/physical_systems/converters/2QC.rst","parts/physical_systems/converters/4QC.rst","parts/physical_systems/converters/B6C.rst","parts/physical_systems/converters/DoubleConv.rst","parts/physical_systems/converters/NoConv.rst","parts/physical_systems/converters/converter.rst","parts/physical_systems/electric_motors.rst","parts/physical_systems/electric_motors/dc_base.rst","parts/physical_systems/electric_motors/dfim.rst","parts/physical_systems/electric_motors/electric_motor.rst","parts/physical_systems/electric_motors/extex.rst","parts/physical_systems/electric_motors/induction_base.rst","parts/physical_systems/electric_motors/permex.rst","parts/physical_systems/electric_motors/pmsm.rst","parts/physical_systems/electric_motors/scim.rst","parts/physical_systems/electric_motors/series.rst","parts/physical_systems/electric_motors/shunt.rst","parts/physical_systems/electric_motors/synchronous_base.rst","parts/physical_systems/electric_motors/synrm.rst","parts/physical_systems/electric_motors/three_phase_base.rst","parts/physical_systems/mechanical_loads/const_speed_load.rst","parts/physical_systems/mechanical_loads/ext_speed_load.rst","parts/physical_systems/mechanical_loads/mechanical_load.rst","parts/physical_systems/mechanical_loads/polystatic.rst","parts/physical_systems/noise_generators/gaussian_white_noise.rst","parts/physical_systems/noise_generators/noise_generator.rst","parts/physical_systems/ode_solvers/euler.rst","parts/physical_systems/ode_solvers/ode_solver.rst","parts/physical_systems/ode_solvers/scipy_ode.rst","parts/physical_systems/ode_solvers/scipy_odeint.rst","parts/physical_systems/ode_solvers/scipy_solve_ivp.rst","parts/physical_systems/physical_system.rst","parts/physical_systems/scml_system.rst","parts/physical_systems/voltage_supplies/ac_1_phase_supply.rst","parts/physical_systems/voltage_supplies/ac_3_phase_supply.rst","parts/physical_systems/voltage_supplies/ideal_voltage_supply.rst","parts/physical_systems/voltage_supplies/rc_voltage_supply.rst","parts/physical_systems/voltage_supplies/voltage_supply.rst","parts/readme.rst","parts/reference_generators/const_reference_generator.rst","parts/reference_generators/multiple_ref_generator.rst","parts/reference_generators/reference_generator.rst","parts/reference_generators/sawtooth_reference_generator.rst","parts/reference_generators/sinusoidal_reference_generator.rst","parts/reference_generators/step_reference_generator.rst","parts/reference_generators/subepisoded_reference_generator.rst","parts/reference_generators/switched_reference_generator.rst","parts/reference_generators/triangular_reference_generator.rst","parts/reference_generators/wiener_process_reference_generator.rst","parts/reference_generators/zero_reference_generator.rst","parts/reward_functions/reward_function.rst","parts/reward_functions/weighted_sum_of_errors.rst","parts/technical_models.rst","parts/technicalbackground.rst","parts/utils.rst","parts/visualizations/console_printer.rst","parts/visualizations/motor_dashboard.rst","parts/visualizations/motor_dashboard_plots/action_plot.rst","parts/visualizations/motor_dashboard_plots/cumulative_constraint_violation_plot.rst","parts/visualizations/motor_dashboard_plots/episode_length_plot.rst","parts/visualizations/motor_dashboard_plots/episode_plots.rst","parts/visualizations/motor_dashboard_plots/mean_episode_reward_plot.rst","parts/visualizations/motor_dashboard_plots/reward_plot.rst","parts/visualizations/motor_dashboard_plots/state_plot.rst","parts/visualizations/motor_dashboard_plots/step_plot.rst","parts/visualizations/motor_dashboard_plots/time_plot.rst","parts/visualizations/visualization.rst"],objects:{},objnames:{},objtypes:{},terms:{"113e":[32,78],"438e":78,"45e":[32,78],"79e":[32,78],"abstract":[1,58,82],"b\u00f6cker":[32,79],"case":[0,58,79],"class":[1,3,77,79,82,86,90,91],"default":[1,32,82],"final":0,"function":[1,64,77],"import":[0,1,64,77,82],"kirchg\u00e4ssner":[0,64],"new":[0,78],"short":18,"static":[1,48],"switch":[0,1,64,67,79],"true":82,"while":0,For:[0,1,32,79],I_S:[32,78,79],L_s:[34,40],ODE:[0,1,58],ODEs:0,One:[1,31],R_s:[32,34,39,40,44,79],The:[0,1,29,31,32,51,58,64,78,79,82,86,90,91],Then:64,There:64,These:0,U_S:[32,79],__main__:64,__name__:64,abc:0,about:79,abs:64,absolut:77,academ:0,academia:64,acceler:0,access:0,account:0,accur:0,action:[1,64,79,82],action_plot:82,action_spac:64,add:82,added:51,addit:[0,79],addition:58,additional_plot:82,affili:0,afterward:[32,79],against:0,agent:[0,64],algorithm:[0,64],all:[1,18,31,32,58,64,78,79,82,86,90,91],allow:[0,29,64],alpha:[32,34,40,79],alreadi:0,also:[0,64,79],altern:[0,64,77],among:64,amper:78,analysi:0,analyz:0,angl:[32,79],angular:[0,32,78,79],ani:[0,64],anoth:0,ansi:0,antriebstechnik:[32,79],api:0,appli:[0,82],applic:[0,64],approach:0,arbitrari:[0,29],armatur:[32,78,79],arn:0,around:64,art:0,articl:64,arxiv:64,author:[0,64],automat:77,autoref:0,avail:[0,1,3,31,35,48,51,53,57,63,64,67,76,82,92],averag:0,axi:[32,78,79,86,90,91],balakrishna:0,base:[0,1,3,86,90,91],baselines3:[0,64],basic:[0,1,64,79],becaus:0,bee:58,been:0,befor:79,behavior:0,below:[58,78,79],benchmark:0,besid:0,beta:[32,34,40,79],better:77,between:[0,32,79],beyond:0,bia:77,bib:0,bibliographi:0,bibtex:64,blob:0,boecker2018a:[32,79],boecker2018b:[32,79],book:[0,64],both:[0,64,79],box:[0,1],bridg:79,build:79,built:[0,64],cad:0,cage:[0,1,18,35,64],calcul:[0,64],call:79,callback:1,can:[0,1,29,32,58,64,78,79,82],capabl:0,categori:0,centerpiec:0,chang:0,character:0,charg:0,chiasson2005:[32,79],chiasson:[32,79],chollet2015:0,choos:78,chosen:0,circuit:[32,78,79],cite:64,classic:[0,1,64],clone:64,close:[0,64],code:0,colab:0,colaboratori:64,com:[0,64],combin:0,come:0,command:64,commerci:0,common:82,commonli:0,compar:1,comparison:0,complement:0,complet:0,compon:0,comprehens:0,comput:[32,79],concept:0,concern:0,configur:1,conncet:[32,79],connect:[0,79],consequ:0,consid:[0,32,79],consist:[58,64,79],consol:[1,92],constant:[0,1,48,67,78,79],constraint:[1,82],construct:[0,64],constructor:82,consum:0,contain:[78,79],context:0,continu:[0,1,18,64,79],continuousdynamicallyaveragedconvert:31,control:[0,1,31,32,64,77,79],conveni:0,convers:0,convert:[0,1,57,64,78],cookbook:64,coordin:[0,32,79],core:1,correspond:0,counterpart:64,coupl:0,cours:0,cov:64,cover:0,coverag:64,creat:[0,82],cumul:[1,82,90],current:[0,32,77,78,79],cycl:64,dashboard:[1,92],data:[32,78,79,86,90,91],date:0,dcextexcont:18,dcextexdisc:18,dcmotorsystem:58,dcpermexcont:18,dcpermexdisc:18,dcseriescont:18,dcseriesdisc:18,dcshuntcont:18,dcshuntdisc:18,dead:79,deal:79,decis:64,deep:64,defin:[0,58,77,82],demand:0,demonstr:[0,64],depart:0,depend:0,depict:0,deploy:0,deriv:[31,32,58,79],describ:[0,78,79],descript:[32,78,79],design:0,desir:0,detail:[32,79],determinist:64,develop:[0,1],dfim:64,dfimcont:18,dfimdisc:18,diagnos:0,diagram:[32,79],dict:77,differ:[0,1,32,79],differenti:[0,79],difficulti:0,dimension:0,direct:[0,79],directli:[0,78,82],discret:[0,1,18,64,79],discreteconvert:31,distanc:0,divid:31,doc:64,docu:0,document:0,doe:0,doi:64,domain:0,done:64,doubli:[1,18,35,64],downstream:0,drive:[0,32,64,79],driven:[0,64],dspace:0,due:[0,79],duti:64,dynam:0,each:[1,32,58,79],earli:0,easi:[0,32,64,79],easili:0,educ:0,effect:[78,79],effort:0,einforc:64,either:[0,64,78],electr:[0,58,78,79],electromagneticwork:0,electron:[0,1,58,64,78,79],elektrisch:[32,79],element:0,elementari:29,enabl:0,encourag:0,energi:0,engin:[0,64],entri:64,env:[64,77,82],environ:[0,1,77,78,82],environment_featur:0,episod:[1,82],episodeplot:86,epsilon:[32,79],equat:[0,32,78,79],equip:0,error:[1,76],euler:[1,53],everi:0,evid:0,exampl:[32,64,79],excit:[1,18,29,32,35],execut:64,exemplari:0,exhibit:0,exist:0,expedi:0,experi:64,expert:0,explan:79,extern:[1,18,29,32,35,48],extex:78,facet:0,factor:78,fals:[77,82],featur:64,fed:[1,18,35,64],feedback:0,femm:0,few:0,field:0,fig:0,figur:[32,58,79],file:1,finit:0,first:[32,79],fix:0,flexibl:0,flux:[32,78,79],folder:[0,64],follow:[0,1,64],forc:0,form:[0,53],found:[0,1,32,64,79],four:[0,1,31],frac:[32,34,36,38,39,40,41,42,44,49,53,79],frame:0,framework:[0,64],free:0,freeli:0,frequenc:0,from:[0,31,32,58,64,77,79,82],full:64,further:[1,79,82],furthermor:[0,18,32,78,79],gamma:77,gaussian:[1,51],gem:[0,58,77,82],gem_cookbook:0,gener:[0,1,58,64],geq:79,germani:0,gerrit:0,git:64,github:[0,64],give:64,given:[18,32,79],googl:[0,64],got:58,gradient:64,graduat:0,graphic:0,great:0,grid:0,guid:[1,64],gym:0,gym_electric_motor:[64,77,82],half:79,hardwar:0,has:[0,32,58,78,79],hat:[32,79],have:[0,77,82],heavili:0,henc:0,henri:[32,78],here:[0,82],herein:0,high:[32,79],hoboken:[32,79],howev:0,http:0,hypersim:0,i_E:79,i_a:[32,36,42,77,79],i_a_n:78,i_b:[32,79],i_c:[32,79],i_e:[36,42,77],i_e_n:78,i_n:[32,78],ideal:[0,1,63],ieee:64,ignor:77,implement:[0,58],includ:[0,29,78,79],index:[0,1],induc:0,induct:[0,1,18,32,35,64,78,79],industri:[0,64],inertia:[0,32,78,79],inform:79,initi:[77,78],innov:0,input:[0,32,79],insid:1,inspect:0,inspir:0,instantan:0,instanti:[64,82],institut:0,integr:[0,1,53],intellig:64,interact:64,interest:0,interfac:[0,1,18,58,64],interlock:79,interpret:0,interv:79,introduc:79,introduct:18,investig:0,ipynb:0,its:[0,78],j_load:78,j_rotor:[32,78],joachim:[32,79],john:[32,79],journal:64,just:0,kei:[32,78],kera:[0,64],kickstart:64,knowledg:0,l_a:[36,38,41,42,78,79],l_d:[32,39,44,78,79],l_e:[36,41,42,78,79],l_e_prim:78,l_m:[34,40],l_q:[32,39,44,78,79],l_r:[34,40],languag:0,larg:0,latest:0,lea:[0,64],learn:[0,1,64],lectur:0,length:[1,82],librari:0,licens:0,like:[0,1,82],likewis:0,limit:[0,1,3],line:[32,78,79],linear:[0,78],link:[32,78,79],linkag:78,list:[0,18,64,82],literatur:0,load:[0,1,32,57,64],loop:[0,64],lot:0,low:0,machin:[32,79],magnet:[0,1,18,35,64],magnitud:0,mai:0,main:0,make:[0,64,77,82],margoli:0,master:0,mathbf:53,mathit:[32,38,39,79],mathrm:[0,32,34,36,38,39,40,41,42,44,49,53,79],mathwork:0,matlab:0,maxim:0,maximilian:0,maxwel:0,mean:[1,64,82],meanepisoderewardplot:82,mechan:[0,1,32,58,78,79],meeker:0,method:[0,58],minim:0,minimalist:64,mode:[0,64],model:[0,1,32,64],modul:1,modular:0,moment:[0,32,78,79],momentari:0,monitor:[0,1],more:[0,32,79],most:[0,64,82],mostli:[32,79],motor:[0,29,57,92],motor_dashboard_plot:82,motordashboard:82,motordesignltd:0,motorwizard:0,mpc:64,multi:[1,31],multipl:[1,67,82],must:29,name:[0,79,82],necessari:0,need:[32,79],neg:[0,77],neighbour:0,network:64,neural:64,neurips2019_9015:0,nois:[1,58],nomin:[32,78,79],non:0,norm:77,normed_reward_weight:77,notabl:0,note:64,notebook:[0,64],number:[0,29,32,64,78,79,86,90],numer:0,object:77,observ:0,obtain:64,ode:[1,53],odeint:[1,53],offer:0,ohm:[32,78],oliv:0,omega:[0,32,49,78,79,82],omega_:[32,34,36,39,40,41,42,44,49,78,79],omega_n:78,one:[1,32,78,79],ones:82,onli:[58,64,78,79],opal:0,open:0,openai:[0,64],openmodelica:0,oper:0,optim:0,option:0,orcid:0,order:0,ordinari:0,org:64,orient:0,osmc2020:0,other:[0,32,79],otherwis:77,out:[0,1],output:[0,79],overal:58,own:1,packag:[1,64,79],paderborn:[0,32,79],page:[1,18,64],pai:0,pair:[32,78,79],paper:64,par:0,parallel:79,parameter:0,parametr:78,part:[1,79],partial:0,particular:0,particularli:0,pass:[64,77,78,82],past:0,peak:[32,78,79],perform:[0,32,79],perman:[1,18,35,64],permex:78,phase:[0,1,31,32,35,63,64,78],physic:[0,1,64],pick:64,pip:64,plai:64,plappert2016kerasrl:0,pleas:64,plot:[0,1,82],plug:[0,64],pmsm:[32,64],pmsmcont:18,pmsmdisc:[18,64],point:0,polar:79,pole:[32,78,79],polici:64,polynomi:[1,48,79],popular:0,posit:[32,77,79],possess:0,possibl:79,potenti:0,power:[0,1,58,64,78,79],powerelectronicconvert:31,praneeth:0,predefin:0,predict:0,preprint:64,present:[0,1],preval:0,prime:[36,41,42],prime_:79,printer:[1,92],privat:58,probabl:0,problem:77,procedur:0,process:[0,1,67],produc:[32,79],program:0,project:[0,64],promot:0,proper:0,properti:78,prototyp:0,provid:[0,79],psi:[32,38,39,79],psi_:[34,40],psi_e:78,psi_p:[32,78],purpos:0,pypi:64,python:[0,64],pytorch:0,quad:34,quadrant:[1,31],quadrat:[0,78],quantiti:[32,79],quasi:0,quick:[0,1],quickli:82,quickstart:64,r_a:[36,38,41,42,78,79],r_e:[36,41,42,78,79],r_r:[34,40],r_s:[32,78],rad:78,random:64,rang:[64,77,79],rapid:0,rare:0,readm:1,real:0,receiv:0,recent:0,recommend:64,refer:[1,64],reinforc:[0,1,64],relat:[32,79],releas:64,reluct:[1,18,35,64,79],render:[0,64],repres:[0,79],requir:[0,64],research:0,reset:[0,64],resist:[32,78,79],resourc:0,respect:0,result:[0,32,79],reward:[0,1,64,77,82],reward_funct:77,reward_plot:82,reward_pow:77,reward_weight:77,rewritten:79,rich:64,rl2:64,rms:[32,79],root:[64,77],rotat:[32,79],rotor:[32,78,79],routin:[0,64],same:[32,58,79],sampl:[64,79],sawtooth:[1,67],scenario:0,schenk:0,scientif:0,scilab2020:0,scim:[0,64],scim_exampl:0,scim_ideal_grid_simul:0,scimcont:18,scimdisc:18,scipi:[1,53],scml:[1,57],scml_system:0,script:64,search:1,section:1,seen:58,select:[0,1,82],self:[77,82],seri:[1,18,35],set:[0,78,79],setup:[0,64],shall:64,sheet:[32,78,79],shift:77,shortli:79,show:[0,79],showcas:[0,64],shown:[32,79],shunt:[1,18,35],sigma:[34,40],sign:[49,79],similar:[32,79],simpl:[0,64],simplifi:0,simul:[0,1,58,64],simulink:0,simupi:0,sinc:0,sinusoid:[1,67],snippet:0,softwar:1,solut:0,solv:53,solve_ivp:[1,53],solver:[1,58],sourc:[0,64],space:79,spatial:0,specif:0,specifi:[0,1,79],speed:[0,1,48],sqrt:[32,78,79],squar:[1,3,77],squirrel:[0,1,18,35,64],stabl:[0,64],stai:[0,58],standalon:64,star:[32,79],start:0,state:[0,1,51,64,79,82],state_plot:82,stator:[32,78,79],step:[0,1,64,67,79,82],stepplot:90,strong:0,strongli:0,structur:[0,58,64,79],student:0,style:0,subcompon:58,subconvert:29,subepisod:[1,67],sum:[1,76],sup:[0,32,79],suppli:[0,1,32,57,64,78,79],support:0,surpass:0,symmetr:79,synchron:[0,1,18,35,64],synchronousmotorsystem:58,synrm:[64,79],synrmcont:18,synrmdisc:18,synthesi:0,system:[0,1,53,64,79],t_l:[32,49,79],tag:0,take:0,talent:0,task:0,tau_:[34,40],tau_r:[34,40],technic:58,tensorflow2015:0,tensorflow:0,tensorforc:64,term:78,test:[0,1],text:[32,34,40,79],tfagent:0,thei:[0,58],them:1,therefor:[0,32,58,64,78,79],thereof:0,thi:[0,1,18,32,58,64,79],those:79,three:[0,1,31,32,35,64],time:[0,1,79,82],timeplot:91,titl:[0,64],tnnl:64,todo:3,togeth:64,too:0,toolbox:[0,1,32,58,64,79],topic:0,torqu:[0,32,78,79],torque_n:78,total:49,toward:64,track:77,traction:0,train:[0,1,64],trajectori:0,transact:64,transform:[0,32,78,79],traue:[0,64],triangular:[1,67],tutori:0,two:[0,1,31],type:79,typic:[0,32,64,79],u_E:79,u_a:[32,36,79],u_a_n:78,u_b:[32,79],u_c:[32,79],u_e:36,u_e_n:78,u_l:[32,78,79],u_n:[32,78],u_sup:78,under:64,univers:[0,32,79],upb:[0,64],upon:64,usa:[32,79],usag:0,use:[0,29,64,77,82],used:[0,1,64,79],useful:0,user:[0,58],usual:[0,64],util:[1,64],valid:79,valu:[0,32,78,79],varepsilon_:[34,39,40,44],variabl:[0,51],varieti:0,variou:64,veloc:[0,32,78,79],veo:0,veri:0,vert:[49,79],via:0,view:0,violat:[0,1,77,82],violation_reward:77,visual:[0,1,64,82],volt:78,voltag:[0,1,32,58,64,78],volum:64,wai:[0,64],wallscheid:[0,64],want:0,weight:[1,76],weightedsumoferror:77,well:[0,64],when:0,where:79,which:[0,51],white:[1,51],whitepap:[0,64],wide:0,wiener:[1,67],wilhelm:0,within:0,worldwid:0,wse:77,xco:0,xplore:64,year:64,you:[64,77,82],zero:[1,67]},titles:["Summary","Welcome to gym-electric-motor(GEM)\u2019s documentation!","Callbacks","Constraint Monitor","Constraint Base Class","Limit Constraint","Squared Constraint","Core","Continuous DC Externally Excited Motor Environment","Discrete DC Externally Excited Motor Environment","Continuous DC Permanently Excited Motor Environment","Discrete DC Permanently Excited Motor Environment","Continuous DC Series Motor Environment","Discrete DC Series Motor Environment","Continuous DC Shunt Motor Environment","Discrete DC Shunt Motor Environment","Continuous Doubly Fed Induction Motor Environment","Discrete Doubly Fed Induction Motor Environment","Environments","Continuous Permanent Magnet Synchronous Motor Environment","Discrete Permanent Magnet Synchronous Motor Environment","Continuous Squirrel Cage Induction Motor Environment","Discrete Squirrel Cage Induction Motor Environment","Continuous Synchronous Reluctance Motor Environment","Discrete Synchronous Reluctance Motor Environment","One Quadrant Converters","Two Quadrant Converters","Four Quadrant Converters","Three Phase Converters","Multi Converters","No Converter","Power Electronic Converters","Electric Motors","Base DC Motor","Doubly Fed Induction Motor","Electric Motors","Externally Excited DC Motor","Base Induction Motor","Permanently Excited DC Motor","Permanent Magnet Synchronous Motor","Squirrel Cage Induction Motor","DC Series Motor","DC Shunt Motor","Base Synchronous Motor","Synchronous Reluctance Motor","Base Three Phase Motor","Constant Speed Load","External Speed Load","Mechanical Loads","Polynomial Static Load","Gaussian White Noise Generator","Noise Generators","Euler Solver","ODE-Solvers","scipy.integrate.ode Solver","scipy.integrate.odeint Solver","scipy.integrate.solve_ivp Solver","Physical Systems","Supply Converter Motor Load System (SCML)","AC 1 Phase Supply","AC 3 Phase Supply","Ideal Voltage Supply","RC Voltage Supply","Voltage Supplies","Readme File","Constant Reference Generator","Multiple Reference Generator","Reference Generators","Sawtooth Reference Generator","Sinusoidal Reference Generator","Step Reference Generator","Subepisoded Reference Generator","Switched Reference Generator","Triangular Reference Generator","Wiener Process Reference Generator","Zero Reference Generator","Reward Functions","Weighted Sum of Errors","Technical Models","Technical Background","Utils","Console Printer","Motor Dashboard","Action Plot","Cumulative Constraint Violation Plot","Episode Length Plot","Episode Plot (Abstract)","Mean Episode Reward Plot","Reward Plot","State Plot","Step Plot (Abstract)","Time Plot (Abstract)","Visualization"],titleterms:{"1qc":79,"2qc":79,"4qc":79,"abstract":[86,90,91],"class":[4,31,32,35,46,47,48,49,51,53,57,63,67,76,92],"default":78,"function":76,"static":49,"switch":72,ODE:[34,36,38,39,40,41,42,44,49,53],One:25,action:83,api:[3,4,77,82],architectur:0,averag:31,background:79,base:[4,18,31,32,33,35,37,43,45,48,51,53,57,63,67,76,92],block:64,bridg:28,build:64,cage:[21,22,40,58],callback:2,citat:64,code:[33,34,36,37,38,39,40,41,42,43,44,45],consol:81,constant:[46,65],constraint:[3,4,5,6,84],content:[1,82],continu:[8,10,12,14,16,19,21,23,25,26,27,28,29,31],convert:[25,26,27,28,29,30,31,58,79],core:7,cumul:84,dashboard:82,defin:4,descript:[46,47,49],dictionari:[32,78],discret:[9,11,13,15,17,20,22,24,25,26,27,28,29,31],document:[1,3,4,33,34,36,37,38,39,40,41,42,43,44,45,77],doubli:[16,17,34,58],dynam:31,electr:[1,18,32,34,35,36,38,39,40,41,42,44,64],electron:31,environ:[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,64],episod:[85,86,87],equat:[34,36,38,39,40,41,42,44,49],error:77,euler:52,exampl:0,excit:[8,9,10,11,36,38,78,79],extern:[8,9,36,47,78,79],featur:0,fed:[16,17,34,58],file:64,flow:64,four:27,gaussian:50,gem:[1,64],gener:[50,51,65,66,67,68,69,70,71,72,73,74,75],get:[1,64],guid:[3,77,82],gym:[1,64],how:4,ideal:61,indic:1,induct:[16,17,21,22,34,37,40,58],inform:64,instal:64,integr:[54,55,56],introduct:78,invert:79,length:85,limit:5,load:[46,47,48,49,58,78,79],magnet:[19,20,32,39,79],mean:87,mechan:[48,49],mechanicalload:48,model:[78,79],monitor:3,motor:[1,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,32,33,34,35,36,37,38,39,40,41,42,43,44,45,58,64,78,79,82],multi:29,multipl:66,need:0,nois:[50,51],ode:54,odeint:55,overview:64,own:4,packag:0,paramet:[32,78],perman:[10,11,19,20,32,38,39,78,79],phase:[28,45,59,60,79],physic:57,plot:[83,84,85,86,87,88,89,90,91],pmsm:[78,79],polynomi:49,power:31,printer:81,process:74,pytest:64,quadrant:[25,26,27,79],readm:64,refer:[0,32,65,66,67,68,69,70,71,72,73,74,75,79],relat:0,reluct:[23,24,32,44],reward:[76,87,88],run:64,sawtooth:68,schemat:[34,36,38,39,40,41,42,44],scipi:[54,55,56],scml:58,seri:[12,13,41,78,79],shunt:[14,15,42,78,79],sinusoid:69,softwar:0,solve_ivp:56,solver:[52,53,54,55,56],sourc:79,speed:[46,47],squar:6,squirrel:[21,22,40,58],start:[1,64],state:89,statement:0,step:[70,90],subepisod:71,sum:77,summari:0,suppli:[58,59,60,61,62,63],synchron:[19,20,23,24,32,39,43,44,58,79],synrm:78,system:[57,58],tabl:1,technic:[78,79],test:64,three:[28,45,79],time:91,torqu:[34,36,38,39,40,41,42,44],triangular:73,two:26,unit:64,usag:[3,77,82],util:80,violat:84,visual:92,voltag:[61,62,63,79],weight:77,welcom:1,white:50,wiener:74,your:4,zero:75}})