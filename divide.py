
import os
import shutil

for i in os.listdir('./frames/'):
    print(i)
    if '_b' in i:
        shutil.copyfile('./frames/'+i,'./BirdeyeView/'+i)
    else:
        shutil.copyfile('./frames/'+i,'./DashView/'+i)

print("Finished")