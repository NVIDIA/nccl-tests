import os 
import xlrd
import xlwt
# 设置环境变量
os.environ['LD_LIBRARY_PATH'] = "/home/panlichen/zrk/work/ofccl/build/lib"
os.environ['NCCL_PROTO'] = "Simple"
os.environ['NCCL_ALGO'] = "RING"
# test
# f = os.popen("./nccl/run.sh")
# print(f.readlines())
# 设置超参数
# run
DATE="221222"
runNcclTest = False # 运行nccl测试
collectNcclResult  = True  # 统计nccl测试结果，写入xls
runOfcclTest = False# 运行ofccl测试
collectOfcclResult = True # 统计ofccl测试结果，写入xls

NCCL_ORDER="1"
resultXlsName="result_"+DATA+"_"+NCCL_ORDER+".xls"
n = 2
m = 3 #nccl
w = 2
M = 3 #ofccl
NUM_DEV = 4#设备的卡数，实验用到的卡数写在循环里

# static 
os.system("g++ ./nccl/static_nccl.cpp -o ./nccl/static_nccl.out")
os.system("g++ ./nccl/static_time.cpp -o ./nccl/static_time.out")
os.system("g++ ./ofccl/clear_static_ofccl_time.cpp -o ./ofccl/clear_static_ofccl_time.out")
os.system("g++ ./ofccl/clear_static_ofccl.cpp -o ./ofccl/clear_static_ofccl.out")



table = xlwt.Workbook()
bwSheet = table.add_sheet('bw')
tmSheet = table.add_sheet('time')
cnt  = 0
for MY_NUM_DEV in [2,4]:

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
    NCCL_OUTPUT_BW_PATH=NCCL_RES_DIR+"/result_statics_nccl_"+str(MY_NUM_DEV)+"cards.txt"  
    NCCL_OUTPUT_TIME_PATH=NCCL_RES_DIR+"/result_statics_nccl_"+str(MY_NUM_DEV)+"cards_time.txt"   
    

    if runNcclTest == True:

        os.system("echo  $(date +%F%n%T)>>"+NCCL_OUTPUT_BW_PATH)
        os.system("echo  $(date +%F%n%T)>>"+NCCL_OUTPUT_TIME_PATH)

        for iter in [1,2,3]:
            NCCL_RES_PATH = NCCL_RES_DIR+"/nccl_result_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_m"+str(m)+".txt"
            
            os.system("echo $(date +%F%n%T)>> "+NCCL_RES_PATH)
            for a in ["64" ,"128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M", "512M", "1G"]:
                os.system("../build/all_reduce_perf -b "+str(a)+" -e "+str(a)+" -f 2 -t " +str(MY_NUM_DEV)+" -g 1 -n "+str(n)+" -w "+str(w)+" -c 0 -m "+str(m) +" >>"+ NCCL_RES_PATH)

            os.system("./nccl/static_nccl.out " +NCCL_RES_PATH+" " +NCCL_OUTPUT_BW_PATH+" "+str(MY_NUM_DEV)) 
            os.system("./nccl/static_time.out " +NCCL_RES_PATH+" " +NCCL_OUTPUT_TIME_PATH+" "+str(MY_NUM_DEV)) 
                   
    if collectNcclResult == True :
        # bus
        bwSheet.write(cnt*30,0,str(MY_NUM_DEV)+'卡')

        with open(NCCL_OUTPUT_BW_PATH) as f:
            content = f.read()
        bw = content.split()

        axis_y =  ["64" ,"128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M", "512M", "1G"]
        for a in range(0,25):
            bwSheet.write(2+a+cnt*30,0,axis_y[a])                 
        #
        for k in [0,1,2]:
            bwSheet.write(1+cnt*30,1+k,'nccl-algbw'+str(k))
            for i in range(0,25):
                bwSheet.write(2+i+cnt*30,1+k,bw[i+k*50+2])

            bwSheet.write(1+cnt*30,1+15+k,'nccl-busbw'+str(k))
            for i in range(0,25):
                bwSheet.write(2+i+cnt*30,1+15+k,bw[i+k*50+25+2])
        # avg
        bwSheet.write(1+cnt*30, 4, 'avg-algbw')
        bwSheet.write(1+cnt*30, 19, 'avg-busbw')
        for i in range(0,25):
            bwSheet.write(2+i+cnt*30, 4, xlwt.Formula('SUM(B'+str(2+i+cnt*30+1)+',C'+str(2+i+cnt*30+1)+',D'+str(2+i+cnt*30+1)+')/3') )
            bwSheet.write(2+i+cnt*30, 19, xlwt.Formula('SUM(Q'+str(2+i+cnt*30+1)+',R'+str(2+i+cnt*30+1)+',S'+str(2+i+cnt*30+1)+')/3')) 
        
        # time  
        with open(NCCL_OUTPUT_TIME_PATH) as f2:
            content2 = f2.read()
        times = content2.split()

        tmSheet.write(cnt*30,0,str(MY_NUM_DEV)+'卡')
        for a in range(0,25):
            tmSheet.write(2+a+cnt*30,0,axis_y[a])
        for k in [0,1,2]:
            tmSheet.write(1+cnt*30,1+k,'nccl-'+str(k))
            for i in range(0,25):
                tmSheet.write(2+i+cnt*30,1+k,times[i+k*25+2])
        # avg 
        tmSheet.write(1+cnt*30, 4, 'avg-nccl')
        for i in range(0,25):
            tmSheet.write(2+i+cnt*30, 4, xlwt.Formula('SUM(B'+str(2+i+cnt*30+1)+',C'+str(2+i+cnt*30+1)+',D'+str(2+i+cnt*30+1)+')/3') )
        

    #OFCCL      
    # 创建存放实验结果的文件夹
    OFCCL_RES_DIR ="./ofccl/test_result_"+DATE+"_"+NCCL_ORDER+"_"+str(MY_NUM_DEV)+"cards"
    if not os.path.exists(OFCCL_RES_DIR):
        os.makedirs(OFCCL_RES_DIR)
    # 统计结果    
    OFCCL_OUTPUT_BW_PATH=OFCCL_RES_DIR+"/result_statics_ofccl_"+str(MY_NUM_DEV)+"cards.txt"  
    OFCCL_OUTPUT_TIME_PATH=OFCCL_RES_DIR+"/result_statics_ofccl_"+str(MY_NUM_DEV)+"cards_time.txt"  

    if runOfcclTest == True: 
        os.system("echo  $(date +%F%n%T)>>"+OFCCL_OUTPUT_BW_PATH)
        os.system("echo  $(date +%F%n%T)>>"+OFCCL_OUTPUT_TIME_PATH)

        for iter in [1,2,3]:
            OFCCL_RES_PATH = OFCCL_RES_DIR+"/ofccl_result_"+str(iter)+"_n"+str(n)+"_w"+str(w)+"_M"+str(M)+".txt"
            
            os.system("echo $(date +%F%n%T)>> "+OFCCL_RES_PATH)
            for a in ["64" ,"128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M", "512M", "1G"]:
                os.system("../build/ofccl_all_reduce_perf  -b "+str(a)+" -e "+str(a)+" -f 2 -t " +str(MY_NUM_DEV)+" -g 1 -n "+str(n)+" -w "+str(w)+" -c 0 -M "+str(M) +" >>"+ OFCCL_RES_PATH)

            os.system("./ofccl/clear_static_ofccl.out " +OFCCL_RES_PATH+" " +OFCCL_OUTPUT_BW_PATH+" "+str(MY_NUM_DEV)) 
            os.system("./ofccl/clear_static_ofccl_time.out " +OFCCL_RES_PATH+" " + OFCCL_OUTPUT_TIME_PATH+" "+str(MY_NUM_DEV)) 

    if collectOfcclResult == True:
        
        with open(OFCCL_OUTPUT_BW_PATH) as f2:
            content2 = f2.read()
        bw = content2.split()
        #bus        
        for k in [0,1,2]:
            bwSheet.write(1+cnt*30,5+k,'ofccl-algbw'+str(k))
            for i in range(0,25):
                bwSheet.write(2+i+cnt*30,5+k,bw[i+k*50+2])

            bwSheet.write(1+cnt*30,5+15+k,'ofccl-busbw'+str(k))
            for i in range(0,25):
                bwSheet.write(2+i+cnt*30,5+15+k,bw[i+k*50+25+2])
        # avg
        bwSheet.write(1+cnt*30, 4+4, 'avg-algbw')
        bwSheet.write(1+cnt*30, 19+4, 'avg-busbw')
        for i in range(0,25):
            bwSheet.write(2+i+cnt*30, 4+4, xlwt.Formula('SUM(F'+str(2+i+cnt*30+1)+',G'+str(2+i+cnt*30+1)+',H'+str(2+i+cnt*30+1)+')/3') )
            bwSheet.write(2+i+cnt*30, 19+4, xlwt.Formula('SUM(U'+str(2+i+cnt*30+1)+',V'+str(2+i+cnt*30+1)+',W'+str(2+i+cnt*30+1)+')/3')) 
        
        # time  
        with open(OFCCL_OUTPUT_TIME_PATH) as f2:
            content2 = f2.read()
        times = content2.split()

        for k in [0,1,2]:
            tmSheet.write(1+cnt*30,5+k,'OFccl-'+str(k))
            for i in range(0,25):
                tmSheet.write(2+i+cnt*30,5+k,times[i+k*25+2])
        # avg 
        tmSheet.write(1+cnt*30, 4+4, 'avg-OFCCL')
        for i in range(0,25):
            tmSheet.write(2+i+cnt*30, 4+4, xlwt.Formula('SUM(F'+str(2+i+cnt*30+1)+',G'+str(2+i+cnt*30+1)+',H'+str(2+i+cnt*30+1)+')/3') )

    if collectNcclResult and collectOfcclResult:
        bwSheet.write(1+cnt*30, 9, '(ofccl-nccl)/nccl')
        bwSheet.write(1+cnt*30, 24, '(ofccl-nccl)/nccl')
        tmSheet.write(1+cnt*30, 9, '(ofccl-nccl)/nccl')
        for i in range(0,25):
            bwSheet.write(2+i+cnt*30, 9, xlwt.Formula('(I'+str(2+i+cnt*30+1)+'-E'+str(2+i+cnt*30+1)+')/E'+str(2+i+cnt*30+1)) )
            bwSheet.write(2+i+cnt*30, 24, xlwt.Formula('(X'+str(2+i+cnt*30+1)+'-T'+str(2+i+cnt*30+1)+')/T'+str(2+i+cnt*30+1) ))
            tmSheet.write(2+i+cnt*30, 9, xlwt.Formula('(I'+str(2+i+cnt*30+1)+'-E'+str(2+i+cnt*30+1)+')/E'+str(2+i+cnt*30+1) ) )

    cnt = cnt+1

# 保存 excel
if collectNcclResult or collectOfcclResult:
    table.save(resultXlsName)