#include"bits/stdc++.h"
#include <string>
using namespace std;
int main(int argc,char* argv[]){
    
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);

    int ranks = *(argv[3]) - '0';
    string str;
    stringstream ss;
    vector<string> a;
    vector<string> b;
    string line;
    // time
    getline(cin,line);

    for(int t =0;t < 25;t++){
        for(int i = 0;i < (7+ranks);i++)
            getline(cin,line);
        
        for(int i =0;i < 5;i++)
            cin >> str;

        a.push_back(str);
       
        for(int i = 0;i < 4;i++)
            getline(cin,line);        
        
    }

    for(int i=0;i<a.size();i++)
        cout << a[i] <<endl;

    
    cout<<endl<<endl;
}