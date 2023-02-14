"""
We use the module of the NIDAQ acquisition card. 
We choose the frequency and the time of acquisition 
then we record 5 excels one after the other containing 
the data of time and tension which we will then convert 
into csv to exploit them.
"""

import nidaqmx
import matplotlib.pyplot as plt
from nidaqmx.constants import TerminalConfiguration
import numpy as np
import pandas as pd
frequence=10000
T = 10 #s
numberOfSamples=T*frequence
Signal1 = np.zeros((numberOfSamples,2)) * np.nan
Signal2 = np.zeros((numberOfSamples,2)) * np.nan
Signal3 = np.zeros((numberOfSamples,2)) * np.nan
Signal4 = np.zeros((numberOfSamples,2)) * np.nan
Signal5 = np.zeros((numberOfSamples,2)) * np.nan
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev2/ai5",terminal_config=TerminalConfiguration.RSE)
task.timing.cfg_samp_clk_timing((frequence), source='', active_edge=nidaqmx.constants.Edge.RISING, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=numberOfSamples*5)
task.start()
j=0
while j<5 :
    if j==0:
        value = task.read(numberOfSamples)
        #print(value)
        i=1
    while i<numberOfSamples :
        val = value[i]
        Signal1[i,0] = i/frequence
        Signal1[i,1] = val
        i=i+1
    print('SIGNAL 1 OK')
    if j==1:
        value = task.read(numberOfSamples)
        i=1
    while i<numberOfSamples :
        val = value[i]
        Signal2[i,0] = i/frequence
        Signal2[i,1] = val
        i=i+1
    print('SIGNAL 2 OK')
    if j==2:
        value = task.read(numberOfSamples)
        i=1
    while i<numberOfSamples :
        val = value[i]
        Signal3[i,0] = i/frequence
        Signal3[i,1] = val
        i=i+1
    print('SIGNAL 3 OK')
    if j==3:
        value = task.read(numberOfSamples)
        i=1
    while i<numberOfSamples :
        val = value[i]
        Signal4[i,0] = i/frequence
        Signal4[i,1] = val
        i=i+1
    print('SIGNAL 4 OK')
    if j==4:
        value = task.read(numberOfSamples)
        i=1
    while i<numberOfSamples :
        val = value[i]
        Signal5[i,0] = i/frequence
        Signal5[i,1] = val
        i=i+1
    print('SIGNAL 5 OK')
    j=j+1
df1=pd.DataFrame(Signal1)
df1.to_excel('signal1.xlsx')
print('Recording 1 OK')
df2=pd.DataFrame(Signal2)
df2.to_excel('signal2.xlsx')
print('Recording 2 OK')
df3=pd.DataFrame(Signal3)
df3.to_excel('signal3.xlsx')
print('Recording 3 OK')
df4=pd.DataFrame(Signal4)
df4.to_excel('signal4.xlsx')
print('Recording 4 OK')
df5=pd.DataFrame(Signal5)
df5.to_excel('signal5.xlsx')
print('Recording 5 OK')
plt.figure(1)
plt.plot(Signal1[:,0],Signal1[:,1],Signal2[:,0],Signal2[:,1],Signal3[:,0],Signal3[:,1],Signal4[:,0],Signal4[:,1],Signal5[:,0],Signal5[:,1])
plt.show()
task.stop
task.close()