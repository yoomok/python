//사진파일 이름으로 분류하기

import os
import shutil

for i in os.listdir('./frames/'):
    print(i)
    if '_b' in i:
        shutil.copyfile('./frames/'+i,'./BirdeyeView/'+i)
    else:
        shutil.copyfile('./frames/'+i,'./DashView/'+i)

print("Finished")
