import os 

import xlwt
# 设置字体大小
style = xlwt.XFStyle()
font = xlwt.Font()
font.height = 20*16
style.font = font
# 设置环境变量
#os.environ['LD_LIBRARY_PATH'] = "/home/panlichen/work2/ofccl/build/lib"
os.environ['LD_LIBRARY_PATH'] = "/home/panlichen/zrk/work/ofccl/build/lib"
os.environ['NCCL_PROTO'] = "Simple"
os.environ['NCCL_ALGO'] = "RING"

os.environ['TRAVERSE_TIMES'] = "10"
os.environ['TOLERANT_UNPROGRESSED_CNT'] = "10000"
os.environ['BASE_CTX_SWITCH_THRESHOLD'] = "80"
os.environ['BOUNS_SWITCH_4_PROCESSED_COLL'] = "0"
os.environ['DEV_TRY_ROUND'] = "10"

# 设置超参数
runNcclTest = True # 运行nccl测试,仅输出原始结果
staticNccl = True # 运行统计，输出中间结果
collectNcclResult  = True# 收集nccl测试结果，写入xls


runOfcclTest = True# 运行ofccl测试
staticOfccl = True # 运行统计，输出中间结果
staticOfcclExtral = True# 对ofccl的额外输出进行统计
collectOfcclResult = True# 收集ofccl测试结果，写入xls

buffer_sizes = ["64", "128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M", "512M", "1G"]

TINY_TEST = 0

DATE="230118"
NCCL_ORDER="1"
host=os.environ.get("HOST")
n = 5
m = 1 #nccl
w = 2
M = 1 #ofccl
if host=="oneflow-15" or host=="oneflow-16":
    NUM_DEV = 4#设备的总卡数，实验用到的卡数写在循环里
    ncards = [2,4]
else:
    NUM_DEV = 8
    ncards = [2,4,8]

if TINY_TEST == 1:
    runNcclTest = False # 运行nccl测试,仅输出原始结果
    staticNccl = False # 运行统计，输出中间结果
    collectNcclResult  = False # 收集nccl测试结果，写入xls
    ncards = [2]
    # buffer_sizes = ["64", "128", "256", "512", "1K"]
NCCL_TIER=[1,2,3]
OFCCL_ITER=[1,2,3,4,5,6]
resultXlsName=host+"_"+DATE+"_"+NCCL_ORDER+"_M"+str(m)+"n"+str(n)+"w"+str(w)+".xls"

# static 
os.system("g++ ./nccl/static_nccl.cpp -o ./nccl/static_nccl.out")
os.system("g++ ./nccl/static_time.cpp -o ./nccl/static_time.out")
os.system("g++ ./ofccl/static_ofccl_time.cpp -o ./ofccl/static_ofccl_time.out")
os.system("g++ ./ofccl/static_ofccl_bw.cpp -o ./ofccl/static_ofccl_bw.out")
os.system("g++ ./ofccl/static_ofccl_QE.cpp -o ./ofccl/static_ofccl_QE.out")
os.system("g++ ./ofccl/static_ofccl_QE_ori.cpp -o ./ofccl/static_ofccl_QE_ori.out")
os.system("g++ ./ofccl/static_ofccl_totalCnt.cpp -o ./ofccl/static_ofccl_totalCnt.out")
os.system("g++ ./ofccl/static_ofccl_bw_order.cpp -o ./ofccl/static_ofccl_bw_order.out ")
os.system("g++ ./ofccl/static_ofccl_tm_order.cpp -o ./ofccl/static_ofccl_tm_order.out ")
AR = {}
AG = {}
B = {}
R = {}
RS = {}

table = xlwt.Workbook()
AR['bwSheet'] = table.add_sheet('allReduce_bw')
AR['tmSheet'] = table.add_sheet('allReduce_time')
AR['cntSheet'] = table.add_sheet('allReduce_totalCnt')
AR['run'] = "../build/all_reduce_perf"
AR['runOfccl'] = "../build/ofccl_all_reduce_perf"

AG['bwSheet'] = table.add_sheet('allGather_bw')
AG['tmSheet'] = table.add_sheet('allGather_time')
AG['cntSheet'] = table.add_sheet('allGather_totalCnt')
AG['run'] = "../build/all_gather_perf"
AG['runOfccl'] = "../build/ofccl_all_gather_perf"

B['bwSheet'] = table.add_sheet('broadcast_bw')
B['tmSheet'] = table.add_sheet('broadcast_time')
B['cntSheet'] = table.add_sheet('broadcast_totalCnt')
B['run'] = "../build/broadcast_perf"
B['runOfccl']="../build/ofccl_broadcast_perf"

R['bwSheet'] = table.add_sheet('reduce_bw')
R['tmSheet'] = table.add_sheet('reduce_time')
R['cntSheet'] = table.add_sheet('reduce_totalCnt')
R['run'] = "../build/reduce_perf"
R['runOfccl']= "../build/ofccl_reduce_perf"

RS['bwSheet'] = table.add_sheet('reduceScatter_bw')
RS['tmSheet'] = table.add_sheet('reduceScatter_time')
RS['cntSheet'] = table.add_sheet('reduceScatter_totalCnt')
RS['run'] = "../build/reduce_scatter_perf"
RS['runOfccl'] = "../build/ofccl_reduce_scatter_perf"
# 列宽
# for i in range(30):
#     AR['bwSheet'].col(i).width = 13 * 256
#     AR_tmSheet.col(i).width = 16 * 256

cnt  = 0
for MY_NUM_DEV in ncards:

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    if MY_NUM_DEV == 4 and NUM_DEV == 8:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,4,5"
    # nccl
    # 创建存放实验结果的文件夹
    NCCL_RES_DIR ="./nccl/test_result_"+DATE+"_"+NCCL_ORDER+"_"+str(MY_NUM_DEV)+"cards"
    if not os.path.exists(NCCL_RES_DIR):
        os.makedirs(NCCL_RES_DIR)
    # 统计结果  
    # allReduce  
    AR['nccl_bw_path']=NCCL_RES_DIR+"/result_nccl_allReduce_"+str(MY_NUM_DEV)+"cards.txt"  
    AR['nccl_time_path']=NCCL_RES_DIR+"/result_nccl_allReduce_"+str(MY_NUM_DEV)+"cards_time.txt"   
    # allGather
    AG['nccl_bw_path']=NCCL_RES_DIR+"/result_nccl_allGather_"+str(MY_NUM_DEV)+"cards.txt"  
    AG['nccl_time_path']=NCCL_RES_DIR+"/result_nccl_allGather_"+str(MY_NUM_DEV)+"cards_time.txt"   
    # broadcast
    B['nccl_bw_path']=NCCL_RES_DIR+"/result_nccl_broadcast_"+str(MY_NUM_DEV)+"cards.txt"  
    B['nccl_time_path']=NCCL_RES_DIR+"/result_nccl_broadcast_"+str(MY_NUM_DEV)+"cards_time.txt"   
    # reduce
    R['nccl_bw_path']=NCCL_RES_DIR+"/result_nccl_reduce_"+str(MY_NUM_DEV)+"cards.txt"  
    R['nccl_time_path']=NCCL_RES_DIR+"/result_nccl_reduce_"+str(MY_NUM_DEV)+"cards_time.txt"   
    # reduceScatter
    RS['nccl_bw_path']=NCCL_RES_DIR+"/result_nccl_reduceScatter_"+str(MY_NUM_DEV)+"cards.txt"  
    RS['nccl_time_path']=NCCL_RES_DIR+"/result_nccl_reduceScatter_"+str(MY_NUM_DEV)+"cards_time.txt"      

    if staticNccl == True:
        for op in [AR,AG,B,R,RS]:
            os.system("echo  $(date +%F%n%T)>>"+op['nccl_bw_path'])
            os.system("echo  $(date +%F%n%T)>>"+op['nccl_time_path'])


    for iter in NCCL_TIER:
        # raw data
        AR['nccl_rawData'] = NCCL_RES_DIR+"/nccl_allReduce_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        AG['nccl_rawData'] = NCCL_RES_DIR+"/nccl_allGather_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        B['nccl_rawData'] = NCCL_RES_DIR+"/nccl_broadcast_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        R['nccl_rawData'] = NCCL_RES_DIR+"/nccl_reduce_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        RS['nccl_rawData'] = NCCL_RES_DIR+"/nccl_reduceScatter_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"

        if runNcclTest:
            for op in [AR,AG,B,R,RS]:
                os.system("echo $(date +%F%n%T)>> "+op['nccl_rawData'])

                for a in buffer_sizes:
                    os.system(op['run']+" -b "+str(a)+" -e "+str(a)+" -f 2 -t " +str(MY_NUM_DEV)+" -g 1 -n "+str(n)+" -w "+str(w)+" -c 0 -m "+str(m) +" >>"+ op['nccl_rawData'])

        if staticNccl:
            for op in [AR,AG,B,R,RS]:    
                os.system("./nccl/static_nccl.out " +op['nccl_rawData'] +" " +op['nccl_bw_path']) 
                os.system("./nccl/static_time.out " +op['nccl_rawData'] +" " +op['nccl_time_path'])

            
                   
    if collectNcclResult :
        for op in [AR,AG,B,R,RS]:
            # bus
            op['bwSheet'].write(cnt*30,0,str(MY_NUM_DEV)+'卡',style)

            with open(op['nccl_bw_path']) as f:
                content = f.read()
            bw = content.split()

            axis_y =  buffer_sizes
            for a in range(0,25):
                op['bwSheet'].write(2+a+cnt*30,0,axis_y[a],style)                 
            #
            for k in [0,1,2]:
                op['bwSheet'].write(1+cnt*30,1+k,'nccl-algbw'+str(k),style)
                for i in range(0,25):
                    op['bwSheet'].write(2+i+cnt*30,1+k,bw[i+k*50+2],style)

                op['bwSheet'].write(1+cnt*30,12+k,'nccl-busbw'+str(k),style)
                for i in range(0,25):
                    op['bwSheet'].write(2+i+cnt*30,12+k,bw[i+k*50+25+2],style)
            # avg
            op['bwSheet'].write(1+cnt*30, 4, 'avg-algbw',style)
            op['bwSheet'].write(1+cnt*30, 15, 'avg-busbw',style)
            for i in range(0,25):
                op['bwSheet'].write(2+i+cnt*30, 4, xlwt.Formula('SUM(B'+str(2+i+cnt*30+1)+',C'+str(2+i+cnt*30+1)+',D'+str(2+i+cnt*30+1)+')/3'),style )
                op['bwSheet'].write(2+i+cnt*30, 15, xlwt.Formula('SUM(M'+str(2+i+cnt*30+1)+',N'+str(2+i+cnt*30+1)+',O'+str(2+i+cnt*30+1)+')/3'),style) 
            
            # time  
            with open(op['nccl_time_path']) as f2:
                content2 = f2.read()
            times = content2.split()

            op['tmSheet'].write(cnt*30,0,str(MY_NUM_DEV)+'卡',style)
            for a in range(0,25):
                op['tmSheet'].write(2+a+cnt*30,0,axis_y[a],style)
            for k in [0,1,2]:
                op['tmSheet'].write(1+cnt*30,1+k,'nccl-'+str(k),style)
                for i in range(0,25):
                    op['tmSheet'].write(2+i+cnt*30,1+k,times[i+k*25+2],style)
            # avg 
            op['tmSheet'].write(1+cnt*30, 4, 'avg-nccl',style)
            for i in range(0,25):
                op['tmSheet'].write(2+i+cnt*30, 4, xlwt.Formula('SUM(B'+str(2+i+cnt*30+1)+',C'+str(2+i+cnt*30+1)+',D'+str(2+i+cnt*30+1)+')/3'), style)
        

    #OFCCL      
    # 创建存放实验结果的文件夹
    OFCCL_RES_DIR ="./ofccl/test_result_"+DATE+"_"+NCCL_ORDER+"_"+str(MY_NUM_DEV)+"cards"
    if not os.path.exists(OFCCL_RES_DIR):
        os.makedirs(OFCCL_RES_DIR)
    # 统计结果    
    AR['ofccl_bw_path']=OFCCL_RES_DIR+"/result_ofccl_allReduce_"+str(MY_NUM_DEV)+"cards.txt"  
    AR['ofccl_bw_order_path']=OFCCL_RES_DIR+"/result_ofccl_allReduce_order_"+str(MY_NUM_DEV)+"cards.txt"  
    AR['ofccl_tm_path']=OFCCL_RES_DIR+"/result_ofccl_allReduce_"+str(MY_NUM_DEV)+"cards_time.txt"  
    AR['ofccl_tm_order_path']=OFCCL_RES_DIR+"/result_ofccl_allReduce_order_"+str(MY_NUM_DEV)+"cards_time.txt"  
    AR['ofccl_qe_path']=OFCCL_RES_DIR+"/result_ofccl_allReduce_"+str(MY_NUM_DEV)+"cards_QE.txt"  
    AR['ofccl_qeOri_path']=OFCCL_RES_DIR+"/result_ofccl_allReduce_"+str(MY_NUM_DEV)+"cards_QE_ori.txt" 
    AR['ofccl_totalCnt_path']=OFCCL_RES_DIR+"/result_ofccl_allReduce_"+str(MY_NUM_DEV)+"cards_totalCnt.txt"

    AG['ofccl_bw_path']=OFCCL_RES_DIR+"/result_ofccl_allGather_"+str(MY_NUM_DEV)+"cards.txt"  
    AG['ofccl_bw_order_path']=OFCCL_RES_DIR+"/result_ofccl_allGather_order_"+str(MY_NUM_DEV)+"cards.txt"
    AG['ofccl_tm_path']=OFCCL_RES_DIR+"/result_ofccl_allGather_"+str(MY_NUM_DEV)+"cards_time.txt"  
    AG['ofccl_tm_order_path']=OFCCL_RES_DIR+"/result_ofccl_allGather_order_"+str(MY_NUM_DEV)+"cards_time.txt"
    AG['ofccl_qe_path']=OFCCL_RES_DIR+"/result_ofccl_allGather_"+str(MY_NUM_DEV)+"cards_QE.txt"  
    AG['ofccl_qeOri_path']=OFCCL_RES_DIR+"/result_ofccl_allGather_"+str(MY_NUM_DEV)+"cards_QE_ori.txt" 
    AG['ofccl_totalCnt_path']=OFCCL_RES_DIR+"/result_ofccl_allGather_"+str(MY_NUM_DEV)+"cards_totalCnt.txt"

    B['ofccl_bw_path']=OFCCL_RES_DIR+"/result_ofccl_broadcast_"+str(MY_NUM_DEV)+"cards.txt" 
    B['ofccl_bw_order_path']=OFCCL_RES_DIR+"/result_ofccl_broadcast_order_"+str(MY_NUM_DEV)+"cards.txt"  
    B['ofccl_tm_path']=OFCCL_RES_DIR+"/result_ofccl_broadcast_"+str(MY_NUM_DEV)+"cards_time.txt"  
    B['ofccl_tm_order_path']=OFCCL_RES_DIR+"/result_ofccl_broadcast_order_"+str(MY_NUM_DEV)+"cards_time.txt"  
    B['ofccl_qe_path']=OFCCL_RES_DIR+"/result_ofccl_broadcast_"+str(MY_NUM_DEV)+"cards_QE.txt"  
    B['ofccl_qeOri_path']=OFCCL_RES_DIR+"/result_ofccl_broadcast_"+str(MY_NUM_DEV)+"cards_QE_ori.txt" 
    B['ofccl_totalCnt_path']=OFCCL_RES_DIR+"/result_ofccl_broadcast_"+str(MY_NUM_DEV)+"cards_totalCnt.txt"

    R['ofccl_bw_path']=OFCCL_RES_DIR+"/result_ofccl_reduce_"+str(MY_NUM_DEV)+"cards.txt"  
    R['ofccl_bw_order_path']=OFCCL_RES_DIR+"/result_ofccl_reduce_order_"+str(MY_NUM_DEV)+"cards.txt" 
    R['ofccl_tm_path']=OFCCL_RES_DIR+"/result_ofccl_reduce_"+str(MY_NUM_DEV)+"cards_time.txt"  
    R['ofccl_tm_order_path']=OFCCL_RES_DIR+"/result_ofccl_reduce_order_"+str(MY_NUM_DEV)+"cards_time.txt" 
    R['ofccl_qe_path']=OFCCL_RES_DIR+"/result_ofccl_reduce_"+str(MY_NUM_DEV)+"cards_QE.txt"  
    R['ofccl_qeOri_path']=OFCCL_RES_DIR+"/result_ofccl_reduce_"+str(MY_NUM_DEV)+"cards_QE_ori.txt" 
    R['ofccl_totalCnt_path']=OFCCL_RES_DIR+"/result_ofccl_reduce_"+str(MY_NUM_DEV)+"cards_totalCnt.txt"

    RS['ofccl_bw_path']=OFCCL_RES_DIR+"/result_ofccl_reduceScatter_"+str(MY_NUM_DEV)+"cards.txt"  
    RS['ofccl_bw_order_path']=OFCCL_RES_DIR+"/result_ofccl_reduceScatter_order_"+str(MY_NUM_DEV)+"cards.txt" 
    RS['ofccl_tm_path']=OFCCL_RES_DIR+"/result_ofccl_reduceScatter_"+str(MY_NUM_DEV)+"cards_time.txt" 
    RS['ofccl_tm_order_path']=OFCCL_RES_DIR+"/result_ofccl_reduceScatter_order_"+str(MY_NUM_DEV)+"cards_time.txt"  
    RS['ofccl_qe_path']=OFCCL_RES_DIR+"/result_ofccl_reduceScatter_"+str(MY_NUM_DEV)+"cards_QE.txt"  
    RS['ofccl_qeOri_path']=OFCCL_RES_DIR+"/result_ofccl_reduceScatter_"+str(MY_NUM_DEV)+"cards_QE_ori.txt" 
    RS['ofccl_totalCnt_path']=OFCCL_RES_DIR+"/result_ofccl_reduceScatter_"+str(MY_NUM_DEV)+"cards_totalCnt.txt"   

    if staticOfccl: 
        for op in [AR,AG,B,R,RS]:
            os.system("echo  $(date +%F%n%T)>>"+op['ofccl_bw_path'])
            os.system("echo  $(date +%F%n%T)>>"+op['ofccl_tm_path'])
            os.system("echo  $(date +%F%n%T)>>"+op['ofccl_bw_order_path'])
            os.system("echo  $(date +%F%n%T)>>"+op['ofccl_tm_order_path'])
    if staticOfcclExtral:
        for op in [AR,AG,B,R,RS]:
            os.system("echo  $(date +%F%n%T)>>"+op['ofccl_qe_path'])
            os.system("echo  $(date +%F%n%T)>>"+op['ofccl_qeOri_path'])
       

    for iter in OFCCL_ITER:
        # raw data
        AR['ofccl_rawData'] =  OFCCL_RES_DIR+"/ofccl_allReduce_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        AG['ofccl_rawData'] =OFCCL_RES_DIR+"/ofccl_allGather_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        B['ofccl_rawData'] = OFCCL_RES_DIR+"/ofccl_broadcast_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        R['ofccl_rawData'] = OFCCL_RES_DIR+"/ofccl_reduce_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
        RS['ofccl_rawData'] = OFCCL_RES_DIR+"/ofccl_reduceScatter_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
       
        if runOfcclTest:
            for op in [AR,AG,B,R,RS]:
                os.system("echo $(date +%F%n%T)>> "+op['ofccl_rawData'])
                for a in buffer_sizes:
                    os.system(op['runOfccl']+"  -b "+str(a)+" -e "+str(a)+" -f 2 -t " +str(MY_NUM_DEV)+" -g 1 -n "+str(n)+" -w "+str(w)+" -c 0 -M "+str(M) +" >>"+ op['ofccl_rawData'])
        if staticOfccl:
            for op in [AR,AG,B,R,RS]:
                os.system("./ofccl/static_ofccl_bw.out " +op['ofccl_rawData']+" " +op['ofccl_bw_path']) 
                os.system("./ofccl/static_ofccl_time.out " +op['ofccl_rawData']+" " + op['ofccl_tm_path'])
        if staticOfcclExtral:
            for op in [AR,AG,B,R,RS]:
                os.system("./ofccl/static_ofccl_QE.out " +op['ofccl_rawData']+" " + op['ofccl_qe_path'])
                os.system("./ofccl/static_ofccl_QE_ori.out " +op['ofccl_rawData']+" " + op['ofccl_qeOri_path'])
                os.system("./ofccl/static_ofccl_totalCnt.out "+op['ofccl_rawData']+" " + op['ofccl_totalCnt_path'])
    if staticOfccl:
        for op in [AR,AG,B,R,RS]:
            os.system("./ofccl/static_ofccl_bw_order.out "+op['ofccl_bw_path']+" "+op['ofccl_bw_order_path']+" "+ str(len(OFCCL_ITER)))
            os.system("./ofccl/static_ofccl_tm_order.out "+op['ofccl_tm_path']+" "+op['ofccl_tm_order_path']+" "+ str(len(OFCCL_ITER)))


    if collectOfcclResult == True:
        #bus width
        for op in [AR,AG,B,R,RS]:
            with open(op['ofccl_bw_order_path']) as f2:
                content2 = f2.read()
            bw = content2.split()
                    
            for k in [0,1,2]:
                op['bwSheet'].write(1+cnt*30,5+k,'ofccl-algbw'+str(k),style)
                for i in range(0,25):
                    op['bwSheet'].write(2+i+cnt*30,5+k,bw[i+k*50+2],style)

                op['bwSheet'].write(1+cnt*30,16+k,'ofccl-busbw'+str(k),style)
                for i in range(0,25):
                    op['bwSheet'].write(2+i+cnt*30,16+k,bw[i+k*50+25+2],style)
            # avg
            op['bwSheet'].write(1+cnt*30,8, 'avg-algbw',style)
            op['bwSheet'].write(1+cnt*30, 19, 'avg-busbw',style)
            for i in range(0,25):
                op['bwSheet'].write(2+i+cnt*30, 8, xlwt.Formula('SUM(F'+str(2+i+cnt*30+1)+',G'+str(2+i+cnt*30+1)+',H'+str(2+i+cnt*30+1)+')/3'), style)
                op['bwSheet'].write(2+i+cnt*30, 19, xlwt.Formula('SUM(Q'+str(2+i+cnt*30+1)+',R'+str(2+i+cnt*30+1)+',S'+str(2+i+cnt*30+1)+')/3'),style) 
            
            # time  
            with open(op['ofccl_tm_order_path']) as f2:
                content2 = f2.read()
            times = content2.split()

            for k in [0,1,2]:
                op['tmSheet'].write(1+cnt*30,5+k,'ofccl-'+str(k),style)
                for i in range(0,25):
                    op['tmSheet'].write(2+i+cnt*30,5+k,times[i+k*25+2],style)
            # avg 
            op['tmSheet'].write(1+cnt*30, 4+4, 'avg-ofccl',style)
            for i in range(0,25):
                op['tmSheet'].write(2+i+cnt*30, 4+4, xlwt.Formula('SUM(F'+str(2+i+cnt*30+1)+',G'+str(2+i+cnt*30+1)+',H'+str(2+i+cnt*30+1)+')/3'), style)

    if collectNcclResult and collectOfcclResult:
        for op in [AR,AG,B,R,RS]:
            op['bwSheet'].write(1+cnt*30, 9, '(ofccl-nccl)/nccl',style)
            op['bwSheet'].write(1+cnt*30, 20, '(ofccl-nccl)/nccl',style)
            op['tmSheet'].write(1+cnt*30, 9, 'ofccl-nccl',style)
            op['tmSheet'].write(1+cnt*30, 10, '(ofccl-nccl)/nccl',style)
            for i in range(0,25):
                op['bwSheet'].write(2+i+cnt*30, 9, xlwt.Formula('(I'+str(2+i+cnt*30+1)+'-E'+str(2+i+cnt*30+1)+')/E'+str(2+i+cnt*30+1)), style)
                op['bwSheet'].write(2+i+cnt*30, 20, xlwt.Formula('(T'+str(2+i+cnt*30+1)+'-P'+str(2+i+cnt*30+1)+')/P'+str(2+i+cnt*30+1) ),style)
                op['tmSheet'].write(2+i+cnt*30, 9, xlwt.Formula('I'+str(2+i+cnt*30+1)+'-E'+str(2+i+cnt*30+1) ),style )
                op['tmSheet'].write(2+i+cnt*30, 10, xlwt.Formula('(I'+str(2+i+cnt*30+1)+'-E'+str(2+i+cnt*30+1)+')/E'+str(2+i+cnt*30+1) ),style )

    # time 各个列的标题
    if staticOfcclExtral:
        for op in [AR,AG,B,R,RS]:
            op['tmSheet'].write(1+cnt*30, 13,'nccl IO',style )
            op['tmSheet'].write(1+cnt*30, 14,'nccl kern',style )
            op['tmSheet'].write(1+cnt*30, 15,'ofccl-nccl kern',style )
            op['tmSheet'].write(1+cnt*30, 16,'before after get sqe',style )
            op['tmSheet'].write(1+cnt*30, 17,'AfterSqe TO BeforeCqe',style )
            op['tmSheet'].write(1+cnt*30, 18,'before after put cqe',style )
            op['tmSheet'].write(1+cnt*30, 19,'beforeSqe TO afterCqe',style )
            op['tmSheet'].write(1+cnt*30, 20,'occl rank0 time',style )
            op['tmSheet'].write(1+cnt*30, 21,'nccl kern ori',style )
            op['tmSheet'].write(1+cnt*30, 27,'before after get sqe ori',style )
            op['tmSheet'].write(1+cnt*30, 33,'AfterSqe TO BeforeCqe ori',style )
            op['tmSheet'].write(1+cnt*30, 39,'before after put cqe ori',style )
            op['tmSheet'].write(1+cnt*30, 45,'beforeSqe TO afterCqe ori',style )

            y = 64
            for i in range(0,25):
                op['tmSheet'].write(2+i+cnt*30,12,y,style)
                y = y*2    

            with open(op['ofccl_qe_path']) as f3:
                content3 = f3.read()
            times = content3.split()
            with open(op['ofccl_qeOri_path']) as f4:
                content4 = f4.read()
            times4 = content4.split()
            for i in range(0,25):
                op['tmSheet'].write(2+cnt*30+i, 13, xlwt.Formula('E'+str(3+i+cnt*30)+'-O'+str(3+i+cnt*30) ),style )
                op['tmSheet'].write(2+cnt*30+i, 14, xlwt.Formula('AVERAGEA(V'+str(3+i+cnt*30)+':Z'+str(3+i+cnt*30)+' )' ),style )
                op['tmSheet'].write(2+cnt*30+i, 15, xlwt.Formula('R'+str(3+i+cnt*30)+'-O'+str(3+i+cnt*30) ),style )
                op['tmSheet'].write(2+cnt*30+i,16,times[2+125*cnt+i],style)
                op['tmSheet'].write(2+cnt*30+i,17,times[2+125*cnt+25+i],style)
                op['tmSheet'].write(2+cnt*30+i,18,times[2+125*cnt+50+i],style)
                op['tmSheet'].write(2+cnt*30+i,19,times[2+125*cnt+75+i],style)
                op['tmSheet'].write(2+cnt*30+i,20,times[2+125*cnt+100+i],style)
                for j in range(0,5):
                    op['tmSheet'].write(2+cnt*30+i,27+j,times4[2+500*cnt+i*5+j],style)
                    op['tmSheet'].write(2+cnt*30+i,33+j,times4[2+500*cnt+125+i*5+j],style)
                    op['tmSheet'].write(2+cnt*30+i,39+j,times4[2+500*cnt+250+i*5+j],style)
                    op['tmSheet'].write(2+cnt*30+i,45+j,times4[2+500*cnt+375+i*5+j],style)

            # cntSheet
            op['cntSheet'].write(cnt*30,0,str(MY_NUM_DEV)+'卡',style)
            axis_y =  buffer_sizes
            for a in range(0,25):
                op['cntSheet'].write(2+a+cnt*30,0,axis_y[a],style)

            op['cntSheet'].write(1+cnt*30,1,"totalCtxSaveCnt_avg",style)
            op['cntSheet'].write(1+cnt*30,2,"totalCtxLoadCnt_avg",style)
            op['cntSheet'].write(1+cnt*30,3,"totalProgressed7SwithchCnt_avg",style)
            op['cntSheet'].write(1+cnt*30,4,"totalUnprogressedQuitCnt_avg",style)
            op['cntSheet'].write(1+cnt*30,6,"totalCtxSaveCnt",style)
            op['cntSheet'].write(1+cnt*30,24,"totalCtxLoadCnt",style)
            op['cntSheet'].write(1+cnt*30,42,"totalProgressed7SwithchCnt",style)
            op['cntSheet'].write(1+cnt*30,60,"totalUnprogressedQuitCnt",style)

            with  open(op['ofccl_totalCnt_path']) as f:
                line = f.readline()
                # save
                for i in range(0,25): 
                    numbers = line.split()
                    op['cntSheet'].write(i+2+cnt*30,1,numbers[0])
                    for j in range(1,len(numbers)):
                        op['cntSheet'].write(i+2+cnt*30,5+j,numbers[j])
                    line = f.readline()
                # load
                for i in range(0,25): 
                    numbers = line.split()
                    op['cntSheet'].write(i+2+cnt*30,2,numbers[0])
                    for j in range(1,len(numbers)):
                        op['cntSheet'].write(i+2+cnt*30,23+j,numbers[j])
                    line = f.readline()
                # totalProgressed7SwithchCnt
                for i in range(0,25): 
                    numbers = line.split()
                    op['cntSheet'].write(i+2+cnt*30,3,numbers[0])
                    for j in range(1,len(numbers)):
                        op['cntSheet'].write(i+2+cnt*30,41+j,numbers[j])
                    line = f.readline()
                # totalUnprogressedQuitCnt
                for i in range(0,25): 
                    numbers = line.split()
                    op['cntSheet'].write(i+2+cnt*30,4,numbers[0])
                    for j in range(1,len(numbers)):
                        op['cntSheet'].write(i+2+cnt*30,59+j,numbers[j])
                    line = f.readline()



    cnt = cnt+1

# 保存 excel
if collectNcclResult or collectOfcclResult:
    table.save(resultXlsName)