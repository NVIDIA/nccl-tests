#include"bits/stdc++.h"
#include <string>
using namespace std;
int main(int argc,char* argv[]){
    
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);

    string inputLine;
    vector<string> a;
    vector<string> b;
    string ss="bandwidth";
    string str = "N/A";
    int cnt = 0;
    while(getline(cin, inputLine)){
        if (inputLine.find(str,0)  == -1)
            continue;

        stringstream line;
        line << inputLine;
        string tmp;
        stack<string> ss;
        while(line >> tmp){
            ss.push(tmp);
        }
        ss.pop();
        ss.pop();
        ss.pop();
        a.push_back(ss.top());
        
        if(++cnt == 25)
            break;
    }

    for(auto a1:a)
        cout<<a1<<endl;

    cout <<endl<< endl;
}