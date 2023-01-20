#include"bits/stdc++.h"
using namespace std;
int main(int argc,char* argv[])
{
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);
    int num = *(argv[3]) - '0';


    string time;
    getline(cin, time);
    vector<priority_queue<double,vector<double>,less<double>>> a(25,priority_queue<double,vector<double>,less<double>>());
    vector<priority_queue<double,vector<double>,less<double>>> b(25,priority_queue<double,vector<double>,less<double>>());


    for(int i = 0;i < num;i++){
        for(int j = 0;j < 25;j++){
            double tmp;
            cin>>tmp;
            a[j].push(tmp);
        }
        for(int j = 0;j < 25;j++){
            double tmp;
            cin>>tmp;
            b[j].push(tmp);
        }
    }

    for(int i = 0;i < num;i++){
        for(int j = 0;j < 25;j++){
            double tmp;
            tmp = a[j].top();a[j].pop();
            cout<<tmp<<endl;
        }
        cout<<endl;
        for(int j = 0;j < 25;j++){
            double tmp;
            tmp = b[j].top();b[j].pop();
            cout<<tmp<<endl;
        }
        cout<<endl<<endl;
    }



}