#include"bits/stdc++.h"
#include <sstream>
using namespace std;
int main(int argc,char* argv[]){
    
    
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);
    
    string inputLine;

    vector<double> sqe_ori;
    vector<double> beforeCqe_ori;
    vector<double> putCqe_ori;
    vector<double> afterCqe_ori;
    string bw="bandwidth";

    
    int cnt=0;
    while(getline(cin, inputLine)){
        if(inputLine.find(bw,0) != -1){
            // 判断结束一个输出
            // before after get sqe
            
            if(++cnt == 25)
            break;
        }
        // rank0 time
        int pos = -1;
            // before after get sqe
        if ((pos=inputLine.find("Rank<0> Blk<0> Thrd<0> coll_id = 0, before after get sqe = ",0) ) != -1){
            pos += 58;
            string numbers = inputLine.substr(pos);
            stringstream ss ;
            ss << numbers;
            for(int i = 0;i < 5;i++){
                double tmp;
                ss >> tmp;
                sqe_ori.push_back(tmp);
            }
            continue;
        }
        //AfterSqe TO BeforeCqe
       if ((pos=inputLine.find("AfterSqe TO BeforeCqe = ",0) ) != -1){
            pos += 24;
            string numbers = inputLine.substr(pos);
            stringstream ss ;
            ss << numbers;
            for(int i = 0;i < 5;i++){
                double tmp;
                ss >> tmp;
                if(tmp > 0.00001)
                    beforeCqe_ori.push_back(tmp);
            }
            continue;
        }

        //before after put cqe
        if ((pos=inputLine.find("before after put cqe = ",0) ) != -1){
            pos += 23;
            string numbers = inputLine.substr(pos);
            stringstream ss ;
            ss << numbers;
            for(int i = 0;i < 5;i++){
                double tmp;
                ss >> tmp;
                if(tmp > 0.00001)
                    putCqe_ori.push_back(tmp);
            }
            continue;
        }

        //beforeSqe TO afterCqe 
        if ((pos=inputLine.find("beforeSqe TO afterCqe = ",0) ) != -1){
            pos += 24;
            string numbers = inputLine.substr(pos);
            stringstream ss ;
            ss << numbers;
            for(int i = 0;i < 5;i++){
                double tmp;
                ss >> tmp;
                if(tmp > 0.00001)
                    afterCqe_ori.push_back(tmp);
            }
            continue;
        }
    }

    // before after get sqe
    for(int i = 0;i <25;i++){
        for(int j =0;j < 5;j++)
            cout<<sqe_ori[i*5+j]<<" ";
        cout<<endl;
    }
    cout <<endl<<endl;
    // // AfterSqe TO BeforeCqe
    for(int i = 0;i <25;i++){
        for(int j =0;j < 5;j++)
            cout<<beforeCqe_ori[i*5+j]<<" ";
        cout<<endl;
    }
    cout <<endl<<endl;

    //before after put cqe 
    for(int i = 0;i <25;i++){
        for(int j =0;j < 5;j++)
            cout<<putCqe_ori[i*5+j]<<" ";
        cout<<endl;
    }
    cout <<endl<<endl;
    // beforeSqe TO afterCqe 
    for(int i = 0;i <25;i++){
        for(int j =0;j < 5;j++)
            cout<<afterCqe_ori[i*5+j]<<" ";
        cout<<endl;
    }
    cout <<endl<<endl;

    cout <<endl<<endl<<endl;
}