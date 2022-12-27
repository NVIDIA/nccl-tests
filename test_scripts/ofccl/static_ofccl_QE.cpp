#include"bits/stdc++.h"
#include <sstream>
using namespace std;
int main(int argc,char* argv[]){
    
    
    freopen(argv[1],"r",stdin);
    freopen(argv[2],"a",stdout);
    
    string inputLine;
    vector<string> time;
    vector<double> sqe;
    vector<double> beforeCqe;
    vector<double> putCqe;
    vector<double> afterCqe;
    string bw="bandwidth";

    int cnt = 0;
    double sqe_sum = 0;
    int sqe_cnt = 0;

    double beforeCqe_sum=0;
    int beforeCqe_cnt = 0;

    double putCqe_sum = 0;
    int putCqe_cnt = 0;

    double afterCqe_sum = 0;
    int afterCqe_cnt = 0;

    while(getline(cin, inputLine)){
        if(inputLine.find(bw,0) != -1){
            // 判断结束一个输出
            // before after get sqe
            double sqe_avg = sqe_sum / sqe_cnt;
            sqe.push_back(sqe_avg);
            sqe_sum = 0;
            sqe_cnt =0;
            // AfterSqe TO BeforeCqe
            double beforeCqe_avg = beforeCqe_sum / beforeCqe_cnt;
            beforeCqe.push_back(beforeCqe_avg);
            beforeCqe_sum =0;
            beforeCqe_cnt =0;
            //before after put cqe
            double putCqe_avg = putCqe_sum / putCqe_cnt;
            putCqe.push_back(putCqe_avg);
            putCqe_sum = 0;
            putCqe_cnt = 0;
            //beforeSqe TO afterCqe
            double afterCqe_avg = afterCqe_sum/afterCqe_cnt;
            afterCqe.push_back(afterCqe_avg);
            afterCqe_sum=0;
            afterCqe_cnt=0;

            if(++cnt == 25)
            break;
        }
        // rank0 time
        int pos = -1;
        if ((pos=inputLine.find("time = ",0) ) != -1){
            pos += 7;
            string t="";
            while(inputLine[pos] != ' '){
                t += inputLine[pos];
                pos++;
            }
            time.push_back(t);
            continue;
        }

        // before after get sqe
        if ((pos=inputLine.find("before after get sqe AVG",0) ) != -1){
            pos += 27;
            string t="";
            while(inputLine[pos] != ' '){
                t += inputLine[pos]; 
                pos++;
            }
            stringstream ss;
            double tt;
            ss << t;
            ss >> tt;
            pos=inputLine.find("weight = ",0);
            pos +=9;
            int count = inputLine[pos] - '0';
            sqe_sum += tt * count;
            sqe_cnt += count; 
            continue;
        }
        //AfterSqe TO BeforeCqe
        if ((pos=inputLine.find("AfterSqe TO BeforeCqe AVG",0) ) != -1){
            pos += 28;
            string t="";
            while(inputLine[pos] != ' '){
                t += inputLine[pos]; 
                pos++;
            }
            stringstream ss;
            double tt;
            ss << t;
            ss >> tt;
            pos=inputLine.find("weight = ",0);
            pos +=9;
            int count = inputLine[pos] - '0';
            beforeCqe_sum += tt * count;
            beforeCqe_cnt += count; 
            continue;
        }

        //before after put cqe
        if ((pos=inputLine.find("before after put cqe AVG ",0) ) != -1){
            pos += 27;
            string t="";
            while(inputLine[pos] != ' '){
                t += inputLine[pos]; 
                pos++;
            }
            stringstream ss;
            double tt;
            ss << t;
            ss >> tt;
            pos=inputLine.find("weight = ",0);
            pos +=9;
            int count = inputLine[pos] - '0';
            putCqe_sum += tt * count;
            putCqe_cnt += count; 
            continue;
        }
        //beforeSqe TO afterCqe 
        if ((pos=inputLine.find("beforeSqe TO afterCqe AVG = ",0) ) != -1){
            pos += 28;
            string t="";
            while(inputLine[pos] != ' '){
                t += inputLine[pos]; 
                pos++;
            }
            stringstream ss;
            double tt;
            ss << t;
            ss >> tt;
            pos=inputLine.find("weight = ",0);
            pos +=9;
            int count = inputLine[pos] - '0';
            afterCqe_sum += tt * count;
            afterCqe_cnt += count; 
            continue;
        }
       
        
    }

    // before after get sqe
    for (auto s:sqe){
        cout << s << endl;
    }
    cout <<endl;
    // AfterSqe TO BeforeCqe
    for(auto s:beforeCqe)
        cout << s<<endl;
    cout<<endl;
    //before after put cqe 
    for(auto s:putCqe)
        cout << s<<endl;
    cout<<endl;
    // beforeSqe TO afterCqe 
    for(auto s :afterCqe)
        cout<<s<<endl;
    cout<<endl;

    // occl rank0 time
    for(auto s:time)
        cout<<s<<endl;
    cout << endl<<endl<<endl<<endl;
}