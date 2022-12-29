#include"bits/stdc++.h"
#include <sstream>
using namespace std;
int main(int argc,char* argv[]){
    
    
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);
    
   string inputLine;
    vector<vector<int>> save_ori(25,vector<int>());
    vector<vector<int>> load_ori(25,vector<int>());
    vector<vector<int>> p7s_ori(25,vector<int>());
    vector<vector<int>> quit_ori(25,vector<int>());
    
    vector<double> save_avg;
    vector<double> load_avg;
    vector<double> p7s_avg;
    vector<double> quit_avg;
   
    string bw="bandwidth";

    int cnt=0;
    while(getline(cin, inputLine)){
        if(inputLine.find(bw,0) != -1){
            // 判断结束一个输出
            // save
            double sum = accumulate(begin(save_ori[cnt]), end(save_ori[cnt]), 0);
            double mean =  sum / save_ori[cnt].size();
            save_avg.push_back(mean);
            // load
            sum = accumulate(begin(load_ori[cnt]), end(load_ori[cnt]),0);
            mean = sum / load_ori[cnt].size();
            load_avg.push_back(mean);
            // p7s
            sum = accumulate(begin(p7s_ori[cnt]), end(p7s_ori[cnt]),0);
            mean = sum / p7s_ori[cnt].size();
            p7s_avg.push_back(mean);
            // quit
            sum = accumulate(begin(quit_ori[cnt]), end(quit_ori[cnt]),0);
            mean = sum / quit_ori[cnt].size();
            quit_avg.push_back(mean);

            if(++cnt == 25)
                break;
        }
      
        int pos = 0;
            // save
        while((pos=inputLine.find("totalCtxSaveCnt=",pos) ) != -1){
            pos += 16;
            int number = 0;
            while(inputLine[pos]>='0' &&inputLine[pos]<='9'){
                number = number*10 + (inputLine[pos]-'0');
                pos++;
            }
            save_ori[cnt].push_back(number);
        }
        pos=0;
        while((pos=inputLine.find("totalCtxLoadCnt=",pos) ) != -1){
            pos += 16;
            int number = 0;
            while(inputLine[pos]>='0' &&inputLine[pos]<='9'){
                number = number*10 + (inputLine[pos]-'0');
                pos++;
            }
            load_ori[cnt].push_back(number);
        }

        pos=0;
        while((pos=inputLine.find("totalProgressed7SwithchCnt=",pos) ) != -1){
            pos += 27;
            int number = 0;
            while(inputLine[pos]>='0' &&inputLine[pos]<='9'){
                number = number*10 + (inputLine[pos]-'0');
                pos++;
            }
            p7s_ori[cnt].push_back(number);
        }

        pos=0;
        while((pos=inputLine.find("totalUnprogressedQuitCnt=",pos) ) != -1){
            pos += 25;
            int number = 0;
            while(inputLine[pos]>='0' &&inputLine[pos]<='9'){
                number = number*10 + (inputLine[pos]-'0');
                pos++;
            }
            quit_ori[cnt].push_back(number);
        }

        
    }

    
    for(int i = 0;i < 25;i++){
        cout << save_avg[i]<<" ";
        for(auto num:save_ori[i])
            cout<<num<<" ";
        cout<<endl;
    }


    for(int i =0;i < 25;i++){
        cout<<load_avg[i]<<" ";
        for(auto num:load_ori[i])
            cout<<num<<" ";
        cout<<endl;
    }
    for(int i =0;i < 25;i++){
        cout<<p7s_avg[i]<<" ";
        for(auto num:p7s_ori[i])
            cout<<num<<" ";
        cout<<endl;
    }

    for(int i =0;i < 25;i++){
        cout<<quit_avg[i]<<" ";
        for(auto num:quit_ori[i])
            cout<<num<<" ";
        cout<<endl;
    }
    cout <<endl;
}