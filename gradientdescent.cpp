#include <map>
#include <cmath>
#include <vector>
#include <cstdio>
#include <random>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

const int MAXSTR = 1000000;
const int NORMALIZED = 1 ;
const int N = 61189;
vector<double> maxVAL (N, 1);
vector<double> minVal (N, 1e20);

typedef pair<int,double> Pair;


default_random_engine generator;
uniform_real_distribution<double> distribution (0, 1);
normal_distribution<double> normal (0.0, 1);

map<string, int> wordids;
map<string, int> labelids;


template<class Object>
double fast_product_sum (vector<Object>& A, vector<Object>& B, double totalA)
{
    double sum = 0;
    int j = 0;
    
    while (j < B.size())
    {
        sum = sum + A[B[j].first].second * B[j].second ;
        j ++;
    }
    
    return sum ;
}

template<class Object>
class sparseMatrix{
public:
    int row, col;
    vector<vector<Object> > entries;
    sparseMatrix(int m, int n)
    {
        row = m;
        col = n;
        entries = vector<vector<Object> > (row);
    }
    sparseMatrix(vector<vector<Object> >& M)
    {
        row = M.size ();
        col = M[0].size ();
        entries = M;
    }
    void enter_key (int r, Object key)
    {
        entries[r].push_back ( key );
        return ;
    }
};


template<class Object>
class Matrix{
public:
    int row, col;
    vector<vector<Object> > entries;
    Matrix(int m, int n)
    {
        row = m;
        col = n;
        entries = vector<vector<Object> > (row, vector<Object> (col));
    }
    Matrix(vector<vector<Object> >& M)
    {
        row = M.size ();
        col = M[0].size ();
        entries = M;
    }
    void enter_key (int r, int c, Object key)
    {
        entries[r][c] = key;
        return ;
    }
    Matrix operator+(const Matrix& M)
    {
        Matrix mat (row, col);
        
        for (int i = 0;i < row; i ++)
            for (int j = 0;j < col ;j ++)
                mat.entries[i][j] = entries[i][j] + M.entries[i][j];
        
        return mat;
    }
    Matrix operator-(const Matrix& M)
    {
        Matrix mat (row, col);
        
        for (int i = 0;i < row; i ++)
            for (int j = 0;j < col ;j ++)
                mat.entries[i][j] = entries[i][j] - M.entries[i][j];
        
        return mat;
    }
    Matrix operator*(const Matrix& M)
    {
        Matrix mat (row, M.col);
        
        for (int i = 0;i < row; i ++)
        {
            for (int j = 0;j < M.col; j ++)
            {
                Object val = 0;
                for (int k = 0 ; k < col; k++)
                {
                    val = val + (entries[i][k] * M.entries[k][j]);
                }
                mat.entries[i][j] = val;
            }
        }
        
        return mat;
    }
};


template<class Object>
class fastVector
{
public:
    int len ;
    vector<Object> vec;
    fastVector (int L)
    {
        len = L ;
        vec = vector<Object> (len);
    }
    fastVector (vector<Object> V)
    {
        len = V.size() ;
        vec = V;
    }
    fastVector operator*(const sparseMatrix<Object>& M)
    {
        vector<Object> nextvec (M.row);
        double totalA = 0;
        for (int i = 0;i < len; i ++)
            totalA = totalA + vec[i].second;
        for (int i = 0;i < M.row; i ++)
        {
            vector<Object> rowvec = M.entries[i];
            double val = fast_product_sum<Object> (vec, rowvec, totalA);
            nextvec[i] = Pair (i, val);
        }
        return nextvec;
    }
};

/*
sparseMatrix<Pair> delta (k,m);
sparseMatrix<Pair> X (m, n + 1);
vector<double> Y (m);
Matrix<double> weights (k, n + 1);
Matrix<double> probability (k,m);
*/

double normsq (vector<double>& w)
{
    double norm = 0;
    for (auto weight : w)
        norm = norm + weight * weight;
    
    return norm;
}

//prints out the value of the function being optimized and the document classifications.

void print_log_likelihood (vector<double> Y, Matrix<double> weights, sparseMatrix<Pair> X, Matrix<double> prob, vector<pair<int,int> >& classification, double lambda,
                           vector<fastVector<Pair> > xW)
{
    classification.erase (classification.begin(), classification.end());
    
    int numKlass = weights.row ;
    int countcorrect = 0;
    
    cout << "First few probabilities ";
    
    for (int ell = 0; ell < X.row; ell ++)
    {
        int klass = 0;
        for (int j  = 0; j < numKlass; j ++)
        {
            if (prob.entries[j][ell] > prob.entries[klass][ell])
                klass = j ;
        }
       // cout <<"Example " << ell << " is classified to be " <<klass<<endl;
        classification.push_back (pair<int,int> (ell, klass));
        
        if (ell < 5)
            cout << prob.entries[klass][ell]<<" ";
        if (abs(klass - Y[ell]) < 1e-9)
            countcorrect ++;
    }
    
    printf ("\n Percentage of correct classification, %lf \n", countcorrect * 100.0/ X.row );
    return ;
}

pair<double,double> get_mean_std (vector<double>& v)
{
    double mean = 0;
    
    for (auto val : v)
        mean = mean + val;
    
    mean /= v.size ();
    
    double std = 0;
    
    for (auto val : v)
        std = std + (val - mean) * (val - mean);
    
    std /= v.size ();
    
    std = sqrt (std);
    
    return pair<double,double> (mean, std) ;
}


template<class Object>
void normalize (Matrix<Object>& X)
{
  
    for (int j = 0;j < X.row; j++)
    {
        pair<double, double> ms = get_mean_std (X.entries[j]);
        double mn = 1e50;
        double mx = -1e50;
        
        for (int i = 0;i < X.col; i++)
            mn = min (mn, X.entries[j][i]);
                
        for (int i = 0;i < X.col; i++)
            mx = max (mx, X.entries[j][i]);

        for (int i = 0; i < X.col; i++)
            X.entries[j][i] = (X.entries[j][i] - mn) / (mx-mn);
    }
    
    return ;
}


void gradient_descent (int m, int k, int n, double eta, double lambda, Matrix<double> delta, sparseMatrix<Pair> X, vector<double> Y,
                       Matrix<double> weights, Matrix<double> prob)
{
    int iterations = 100000; //number of iterations of gradient descent
    
    vector<pair<int,int> > classification;
    
    for (int iter = 0; iter < iterations; iter ++)
    {
        int numKlass = weights.row ;
        vector<fastVector<Pair> > xW;
        for (int j = 0;j < numKlass; j ++)
        {
            fastVector<Pair> fastweight (weights.col);
            for (int idx = 0; idx < weights.col; idx ++)
                fastweight.vec[idx] = Pair(idx,weights.entries[j][idx]);
            
            fastVector<Pair> wiXi = fastweight * X ;
            
            xW.push_back (wiXi);
        }
        
        //O(nk)
        //number of ell values is m (the number of training examples)
       
        for (int ell = 0 ; ell < m; ell++)
        {
            int maxID = 0;
            for (int j = 0; j + 1 < numKlass; j++)
                if (xW[j].vec[ell].second > xW[maxID].vec[ell].second)
                    maxID = j;
            
            double denom = 1;
            
            swap (xW[maxID].vec[ell], xW[0].vec[ell]);
            
            for (int j = 1; j + 1 < numKlass; j ++)
                denom = denom + exp (xW[j].vec[ell].second - xW[0].vec[ell].second);
            
            double denom2 = 1.0/exp (xW[0].vec[ell].second);
            
            prob.entries[0][ell] = 1.0 / (denom + denom2);
            
            denom = log (denom);
            
            for (int j = 1; j + 1 < numKlass; j ++)
            {
                denom = denom + (xW[j-1].vec[ell].second - xW[j].vec[ell].second);
                denom2 = 1.0 / (exp(xW[j].vec[ell].second));
                
                prob.entries[j][ell] = 1.0/ (exp(denom) + denom2);
            }
            
            swap (xW[maxID].vec[ell], xW[0].vec[ell]);
            swap (prob.entries[maxID][ell], prob.entries[0][ell]);
          
            double sum = 0;
            for (int j = 0;j + 1< numKlass; j ++)
                sum = sum + prob.entries[j][ell];
            
            prob.entries[numKlass-1][ell] = abs(1 - sum);
        }
        //O(km)
        
        for (int j = 0;j < numKlass; j ++)
        {
            for (int idx = 0; idx < weights.col ; idx ++)
                weights.entries[j][idx] = weights.entries[j][idx] - eta * weights.entries[j][idx] * lambda;
    
            for (int ell = 0; ell < m; ell ++)
            {
                for (int idx = 0; idx < X.entries[ell].size(); idx ++)
                {
                    double Xiell = X.entries[ell][idx].second ;
                    weights.entries[j][ X.entries[ell][idx].first ] = weights.entries[j][ X.entries[ell][idx].first ] + eta * Xiell * (delta.entries[j][ell] - prob.entries[j][ell]) ;
                }
            }
            
            
            //O(km + total valid entry)
        }
        
        cout<<"Weights look like ";
       
        for (int j = 0 ; j < 5; j ++)
            cout<<weights.entries[0][j]<<" ";
        
        for (int j = 0 ; j < 5; j ++)
            cout<<weights.entries[0][weights.col - j - 1]<<" ";
        
        cout << endl;
        
        cout <<"Running iteration "<<iter<<endl;
        print_log_likelihood (Y,weights,X, prob, classification, lambda, xW);
    }
    
    return ;
}


template<class Object>
void print_out (const Matrix<Object>& M)
{
    for (int i = 0;i < M.row; i ++)
    {
        for (int j = 0 ;j < M.col; j ++)
            cout << M.entries[i][j] <<  " ";
        cout << endl ;
    }
    cout << endl ;
    
    return ;
}

void test_code_matrix ()
{
    Matrix<double> M1 (2,2);
    Matrix<double> M2 (2,2);

    for (int i = 0;i < 2; i ++){
        for (int j = 0;j < 2; j ++){
            M1.enter_key (i,j,rand() % 100);
            M2.enter_key (i,j,rand() % 100);
        }
    }
    
    Matrix<double> M (2,2);
    M = M1 * M2 ;
    
    print_out <double> (M1);
    print_out <double> (M2);
    print_out <double> (M);
    
    return ;
}


/*
 
 https://stackoverflow.com/questions/35419882/cost-function-in-logistic-regression-gives-nan-as-a-result
 
 */

template<class Object>
void normalize (sparseMatrix<Object>& X)
{
    for (int ell = 0; ell < X.row; ell++)
    {
        for (int j = 1;j < X.entries[ell].size(); j ++)
            X.entries[ell][j].second = (X.entries[ell][j].second - minVal[X.entries[ell][j].first]) / (maxVAL[X.entries[ell][j].first] - minVal[X.entries[ell][j].first]);
    }
    
    return ;
}



template<class Object, class Object2>
void initialize_weights (sparseMatrix<Object>& X, Matrix<Object2>& weights, vector<double>& Y)
{
    double mean = 0, std = 1;
    
    //for (int j = 0;j < X.entries[0].size(); j ++)
      //  X.entries[0][j].second = (X.entries[0][j].second - mean) / std;
    
      for (int i = 0;i < 20; i ++)
          for (int j = 0;j < X.col; j ++)
              weights.entries[i][j] = distribution (generator);
    
    for (int ell = 0; ell < X.row; ell++)
    {
        mean = 0 ;
        
        for (int j = 0;j < X.entries[ell].size (); j ++)
            mean = mean + X.entries[ell][j].second ;
        
        mean = mean / 61189;
        
        std = 0;
        
        for (int j = 0;j < X.entries[ell].size (); j ++)
            std = std + (X.entries[ell][j].second - mean) * (X.entries[ell][j].second - mean);
        
        int valentries = X.entries[ell].size ();
        
        std = std + (61189 - valentries) * mean * mean;
        std = std / 61189;
        std = sqrt (std);
        
        normal_distribution<double> gaussian (mean, std);
        
        for (int j = 0; j < X.col; j++)
        {
            weights.entries[(int)Y[ell]][j] = gaussian (generator);
        }
    }
    
    return ;
}


template<class Object>
void print_fastVector (fastVector<Object>& weights)
{
    for (int i = 0 ;i < weights.len; i ++)
        cout << weights.vec[i].second << " ";
    
    cout << endl;
    
    return ;
}

void test_code_sparseMatrix ()
{
    int m = 2;
    int n = 10;
    fastVector<Pair> weights (n) ;
    sparseMatrix<Pair> X (2, 10);
    
    X.enter_key (0, Pair (3,1));
    X.enter_key (0, Pair (5,5));
    
    X.enter_key (1, Pair (6,10));
    X.enter_key (1, Pair (2,3));
    
    for (int i = 0;i < n; i ++)
        weights.vec[i] = Pair (i , rand () % 50);
    
    print_fastVector <Pair> (weights);
    
    weights = weights * X ;
    
    print_fastVector <Pair> (weights);
    
    return ;
}

void take_label_input ()
{
    FILE *in = fopen ("newsgrouplabels.txt", "r");
    
    char buffer[50];

    while (fscanf (in, "%s", buffer) != EOF)
    {
        char label [50];
        int id ;
        sscanf (buffer, "%d %s", &id, label);
        
        string s(label);
        
        labelids[s] = id;
    }
    
    fclose (in);
    
    return ;
}

void take_word_input ()
{
    FILE *in = fopen ("vocabulary.txt", "r");
    
    char buffer[500];
    int id = 0; //word ids are 0 based
    
    while (fscanf (in, "%s", buffer) != EOF)
    {
        string word (buffer);
        wordids[word] = id++;
    }
    
    cout << "Dictionary size is " << wordids.size () << endl;
    
    fclose (in);
    
    return ;
}


void take_training_input ()
{
    int k, m , n;
    double eta = 0.01, lambda = 0.01 ;
    
    k = 20; //number of classes in training.csv
    n = 61188;
    m = 0;
    
    FILE *in = fopen ("training.csv","r");
    
    char buffer[MAXSTR];
    vector<Pair> values ;
    vector<vector<Pair> > M;
    vector<int> docIds;
    vector<double> docClass;
    
    while (fgets (buffer, MAXSTR, in) && M.size () < 400){ //take the first 50 documents
        char *token = strtok (buffer, ",");
        int idx = 0;
        while (token)
        {
            int num = atoi (token);
            values.push_back (Pair (idx ++ , num));
            
            token = strtok (NULL, ",");
        }
        
        vector<Pair> columnVec ;
        columnVec.push_back (Pair(0,1.0));
        minVal[0] = 0;
        docIds.push_back (values[0].second);
        for (int j = 1;j + 1 < values.size (); j ++)
        {
            maxVAL[j] = max (maxVAL[j], values[j].second * 1.0);
            minVal[j] = min (minVal[j], values[j].second * 1.0);
          
            if (values[j].second)
                columnVec.push_back (Pair (j,values[j].second));
        }
        
        docClass.push_back (columnVec.back().second);
        M.push_back (columnVec);
        values.erase (values.begin(), values.end());
        
       // cout << idx << endl;
    }
    
    m = M.size ();
    
    /*
    sparseMatrix<Pair> delta (k,m);
    sparseMatrix<Pair> X (m, n + 1);
    vector<double> Y (m);
    Matrix<double> weights (k, n + 1);
    Matrix<double> probability (k,m);
    */
    
    Matrix<double> delta (k,m);
    sparseMatrix<Pair> X (M);
    vector<double> Y (docClass);
    Matrix<double> weights (k, n + 1);
    Matrix<double> probability (k,m);
    
    fclose (in);
    
    normal_distribution<double> gaussian (0.0, (1/lambda));
    
    for (int i = 0;i < k; i ++)
        for (int j = 0;j <= n; j ++)
            weights.entries[i][j] = gaussian (generator);
    
    int totalvalidentries = 0;
    
    for (int i = 0 ;i < m; i ++)
    {
  //      cout <<"Non zero valid entries in row "<<i<< " is "<< X.entries[i].size()<<endl;
        totalvalidentries = totalvalidentries + X.entries[i].size ();
    }
    
    cout << "total number of valid entries is " << totalvalidentries << endl;

    for (int yj = 0;yj < k; yj++)
    {
        for (int ell = 0; ell < m; ell++)
        {
            if (Y[ell] == yj + 1)
                delta.entries[yj][ell] = 1;
            else
                delta.entries[yj][ell] = 0;
        }
    }
    
    //initialize_weights <Pair, double> (X,weights,Y);
    if (NORMALIZED)
        normalize (X);
    
    lambda = lambda;
    
    gradient_descent (m, k, n, eta, lambda, delta, X, Y, weights, probability);
    
    return ;
}


int main()
{
    test_code_matrix ();
    test_code_sparseMatrix ();
    
    take_word_input ();
    take_label_input ();
    take_training_input ();

    return 0;
}
