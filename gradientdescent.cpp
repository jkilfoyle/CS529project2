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


void gradient_descent (int m, int k, int n, double eta, double lambda)
{
    
    
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

void take_wordlabel_input ()
{
    FILE *in = fopen ("vocabulary.txt", "r");
    
    char buffer[50];
    
    while (fscanf (in, "%s", buffer) != EOF)
    {
        char word [50];
        int id ;
        sscanf (buffer, "%s %d", word, &id);
        
        string w (word);
        
        wordids[w] = id;
        
    }
    
    fclose (in);
    
    in = fopen ("newsgrouplabels.txt","r");
    
    while (fscanf (in, "%s", buffer) != EOF)
    {
        char label [50];
        int id ;
        sscanf (buffer, "%d %s", &id, label);
        
        string  s(label);
        
        labelids[s] = id;
    }
    
    fclose (in);
    
    return ;
}


void take_training_input ()
{
    int k, m , n;
    double eta, lambda ;
    
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
    fastVector<double> Y (m);
    Matrix<double> weights (k, n + 1);
    Matrix<double> probability (k,m);
    */
    
    sparseMatrix<Pair> X (M);
    fastVector<double> Y (docClass);
    
    fclose (in);
    
    return ;
}


int main()
{
    test_code_matrix ();
    test_code_sparseMatrix ();
    
    take_wordlabel_input ();
    take_training_input ();

    return 0;
}
