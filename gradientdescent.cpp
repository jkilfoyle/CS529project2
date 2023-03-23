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

typedef pair<int,double> Pair;


map<string, int> wordids;
map<string, int> labelids;


template<class Object>
double fast_product_sum (vector<Object>& A, vector<Object>& B)
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
        for (int i = 0;i < M.row; i ++)
        {
            auto rowvec = M.entries[i];
            double val = fast_product_sum<Object> (vec, rowvec);
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
/*
double get_Xiell (sparseMatrix<Pair> X, int i, int ell)
{
    vector<Pair>::iterator iter = lower_bound (X.entries[ell].begin(), X.entries[ell].end(), Pair (i,0));
    if (iter == X.entries[ell].end())
        return 0;
    return iter -> second ;
}*/


void gradient_descent (int m, int k, int n, double eta, double lambda, Matrix<double> delta, sparseMatrix<Pair> X, vector<double> Y,
                       Matrix<double> weights, Matrix<double> prob)
{
    int iterations = 50; //number of iterations of gradient descent
    
    for (int iter = 0; iter < iterations; iter ++)
    {
        int numKlass = weights.row ;
        vector<fastVector<Pair> > xW;
        for (int j = 0;j < numKlass; j ++)
        {
            cout << "Running class #"<<j << endl;
            
            fastVector<Pair> fastweight (weights.col);
            for (int idx = 0; idx < weights.col; idx ++)
                fastweight.vec[idx] = Pair(idx,weights.entries[j][idx]);
            
            fastVector<Pair> wiXi = fastweight * X ;
            
            xW.push_back (wiXi);
            
            vector<double> prob_y0_X (wiXi.len);
            vector<double> prob_y1_X (wiXi.len);
            
            //number of ell values is m (the number of training examples)
            
            for (int ell = 0;ell < wiXi.len; ell ++)
            {
                prob_y0_X[ell] = 1 / (1 + exp (wiXi.vec[ell].second));
                prob_y1_X[ell] = exp (wiXi.vec[ell].second) / (1 + exp(wiXi.vec[ell].second));
            }
        }
        
        vector<double> denominator (m, 1);
        
        for (int j = 0;j + 1 < numKlass; j++)
        {
            for (int ell = 0; ell < m; ell ++)
                denominator[ell] = denominator[ell] + exp (xW[j].vec[ell].second);
        }
        
        for (int j = 0; j + 1 < numKlass; j ++)
        {
            for (int ell = 0; ell < m; ell++)
                prob.entries[j][ell] = exp (xW[j].vec[ell].second) / denominator[ell];
        }
        
        for (int ell = 0; ell < m; ell++)
            prob.entries[numKlass-1][ell] = 1.0 / denominator[ell];
   
        
        for (int j = 0;j < numKlass; j ++)
        {
            for (int idx = 0; idx < weights.col ; idx ++)
                weights.entries[j][idx] = weights.entries[j][idx] - lambda * eta * weights.entries[j][idx];
            for (int ell = 0; ell < m; ell ++)
            {
                for (int idx = 0; idx < X.entries[ell].size(); idx ++)
                {
                    double Xiell = X.entries[ell][idx].second ;
                    weights.entries[j][ X.entries[ell][idx].first ] = weights.entries[j][ X.entries[ell][idx].first ] + eta * Xiell * (delta.entries[j][ell] - prob.entries[j][ell]);
                }
            }
        }
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
    double eta = 0.1, lambda = 1 ;
    
    k = 20; //number of classes in training.csv
    n = 61188;
    m = 0;
    
    FILE *in = fopen ("training.csv","r");
    
    char buffer[MAXSTR];
    vector<Pair> values ;
    vector<vector<Pair> > M;
    vector<int> docIds;
    vector<double> docClass;
    
    while (fgets (buffer, MAXSTR, in)){
        char *token = strtok (buffer, ",");
        int idx = 0;
        while (token)
        {
            int num = atoi (token);
            values.push_back (Pair (idx ++ , num));
            
            token = strtok (NULL, ",");
        }
        
        vector<Pair> columnVec ;
        columnVec.push_back (Pair(0,1));
        docIds.push_back (values[0].second);
        for (int j = 1;j + 1 < values.size (); j ++)
        {
            if (values[j].second)
                columnVec.push_back (Pair (j+1,values[j].second));
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
    
    for (int i = 0;i < k; i ++)
        for (int j = 0;j <= n; j ++)
            weights.entries[i][j] = 0;
    
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
