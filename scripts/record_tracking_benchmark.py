"""Record two actual Isaac press cycles as separate image and truth streams."""
import argparse
import json
import math
import time
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8765')
    p.add_argument('--seconds',type=int,default=34)
    p.add_argument('--profile',choices=['demo','validation'],default='demo')
    args=p.parse_args()
    def read(path):return json.load(urllib.request.urlopen(args.url+path,timeout=3))
    def control(value):
        request=urllib.request.Request(args.url+'/api/control',data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        urllib.request.urlopen(request,timeout=3).close()
    for i in range(240):
        try:
            m=read('/comparison/comparison.json')
            if m.get('observable'):break
            if i%30==0:print('Waiting:',m.get('message'),flush=True)
        except Exception:
            pass
        time.sleep(.5)
    else:raise RuntimeError('No observable baseline')
    print('Recording initial rest and two real press/release cycles.',flush=True)
    control({'record':True,'us_press':False,'auto':False,'press_mm':0,'x_mm':0})
    try:
        time.sleep(2)
        if args.profile == 'demo':
            control({'us_press':True})
            for i in range(args.seconds):
                time.sleep(1)
                if i%8==0:
                    s=read('/api/status');print('Capture',i,'s; Jmin',s['min_jacobian'],'motion mm',s['max_displacement_mm'],flush=True)
        else:
            started=time.monotonic(); next_report=0
            while time.monotonic()-started<args.seconds:
                t=time.monotonic()-started
                control({'us_press':False,'press_mm':2*(1-math.cos(2*math.pi*t/20)),
                         'x_mm':math.sin(2*math.pi*t/18)})
                if t>=next_report:
                    s=read('/api/status');print('Validation',round(t,1),'s; Jmin',s['min_jacobian'],'motion mm',s['max_displacement_mm'],flush=True)
                    next_report+=8
                time.sleep(.1)
    finally:
        control({'record':False,'us_press':False,'press_mm':0,'x_mm':0})
    print('Recording finished',flush=True)


if __name__=='__main__':main()
