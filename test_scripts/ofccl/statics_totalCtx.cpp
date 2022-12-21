#include"bits/stdc++.h"
using namespace std;
int main(int argc,char* argv[]){
    cout <<"totalCtxSwitchCnt:"<<" " << argv[1]<< " " << argv[2]<<endl;
    
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);
    cout << argv[1]<<" totalCtxSwitchCnt: "<<endl;
    char c;
    int cnt=0;
    int sum=0;
    bool flag = false;
    bool flag2 = false;
    string  a ="totalCtxSwitchCnt=";
    string b="bandwidth";
    while(cin >>c){
        if(c == '!')
        break;
        flag =true;
        flag2 =true;
        for(int i =0;i < a.size();i++){
            if( c != a[i]){
                flag = false;
            }
            if(i < b.size() && c != b[i]){
                flag2 = false;
            }
            if(flag == false && flag2 == false)
                break;
            cin >> c;
        }
        if(flag){
            cnt++;
            int tmp = 0;
            while( c >= '0' && c<= '9'){
                tmp = tmp*10 + c -'0';
                scanf("%c",&c); 
            }
            sum += tmp;
        }
        if(flag2){
           cout << (sum * 1.0)/cnt<<endl;
           cnt = 0;
           sum = 0;
        }
    }
    cout <<endl<<endl;
    cout <<"*************"<<endl;
   
    return 0;
}
