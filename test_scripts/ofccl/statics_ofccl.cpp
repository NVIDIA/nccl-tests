#include"bits/stdc++.h"
#include <sstream>
using namespace std;
int main(int argc,char* argv[]){
    //cout << "bandwidth"<<" "<< argv[1]<<" "<< argv[2]<<endl;
    
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);
     cout << "bandwidth"<<" "<< argv[1]<<" "<< argv[2]<<endl;
    string inputLine;
    vector<double> a;
    vector<double> b;
    string ss="bandwidth";
    string str = "N/A";
    while(getline(cin, inputLine)){
        if (inputLine.find(str,0)  == -1)
            continue;

        stringstream line;
        line << inputLine;
        double tmp;
        line >> tmp;
        line >> tmp;
        a.push_back(tmp);
        line >> tmp;
        b.push_back(tmp);
    }
    cout << argv[1]<<" algbw: "<<endl;
    for(auto a1:a)
        cout<<a1<<endl;
    cout<<"************"<<endl;
    cout << argv[1]<<" busbw: "<<endl;
    for(auto b1:b)
        cout<<b1<<endl;
        
}