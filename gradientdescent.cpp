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
const int NORMALIZED = 1 ;          //swtiches on normalization of columns
const int N = 61189;
vector<double> maxVAL (N, 700);       //normalization constant of each column
vector<double> minVal (N, 0);
double topaccuracy = 0;             //accuracy percentage of classification

typedef pair<int,double> Pair;


//some random variable generators
default_random_engine generator;
uniform_real_distribution<double> distribution (0, 1);
normal_distribution<double> normal (0.0, 1);

//word and label identifiers
map<string, int> wordids;
map<string, int> labelids;


/*
 Computes the inner product between two sparse vectors A and B.
 */
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


/*
 Class describing a sparse Matrix i.e. most of the entries that are zero are discarded.
 */
template<class Object>
class sparseMatrix{
public:
    int row, col;
    vector<vector<Object> > entries;
    sparseMatrix(int m, int n) //constructor
    {
        row = m;
        col = n;
        entries = vector<vector<Object> > (row);
    }
    sparseMatrix(vector<vector<Object> >& M) //constructor with the sparse matrix M.
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

/*
 A matrix class addition, substraction and multiplication operator builtin. Note that
 this is NOT a sparse matrix.
 */
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


/*
 Sparse Vector (i.e. most entries are zero) class that features fast inner product
 and product between itself and a sparseMatrix.
 */
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
    fastVector operator*(const sparseMatrix<Object>& M) //computes the product between vector and a sparse Matrix.
    {
        vector<Object> nextvec (M.row);
        
        for (int i = 0;i < M.row; i ++)
        {
            vector<Object> rowvec = M.entries[i];
            double val = fast_product_sum<Object> (vec, rowvec);
            nextvec[i] = Pair (i, val);
        }
        return nextvec;
    }
};

/*
 
some variable notations with dimensions as specified in assignment description.
 
sparseMatrix<Pair> delta (k,m);
sparseMatrix<Pair> X (m, n + 1);
vector<double> Y (m);
Matrix<double> weights (k, n + 1);
Matrix<double> probability (k,m);
*/

//prints out the percentage of correctly classified instances.

double print_classified_accuracy (vector<double> Y, Matrix<double> weights, sparseMatrix<Pair> X, Matrix<double> prob, vector<pair<int,int> >& classification, double lambda,
                           vector<fastVector<Pair> > xW)
{
    classification.erase (classification.begin(), classification.end());
    
    int numKlass = weights.row ;
    int countcorrect = 0;
    
    cout << "First few probabilities ";
    
    for (int ell = 0; ell < X.row; ell ++)
    {
        int klass = 0;
        
        /*
         Goes over the entire range of classes for a document
         and selects the one with highest probability
         */
        for (int j  = 0; j < numKlass; j ++)
        {
            if (prob.entries[j][ell] > prob.entries[klass][ell])
                klass = j ;
        }
        classification.push_back (pair<int,int> (ell, klass));
        
        
        //some debugging output
        if (ell < 5)
            cout << prob.entries[klass][ell]<<" ";
        if (abs(klass - Y[ell]) < 1e-9)
            countcorrect ++;
    }
    
    //prints out the correct classification percentage or accuracy and then returns it.
    printf ("\n Percentage of correct classification, %lf \n\n\n\n", countcorrect * 100.0/ X.row );
    return countcorrect * 100.0 / X.row;
}


Matrix<double> gradient_descent (int& m, int& k, int& n, double& eta, double& lambda, Matrix<double>& delta, sparseMatrix<Pair>& X, vector<double>& Y,
                       Matrix<double>& weights, Matrix<double>& prob)
{
    int iterations = 100000; //number of iterations of gradient descent
    Matrix<double> bestWeights (weights.entries); //stores the best set of weights found so far.
    
    vector<pair<int,int> > classification; //stores the classification predicted in the last iteration
    
    for (int iter = 0; iter < iterations; iter ++)
    {
        int numKlass = weights.row ; //number of classes
        vector<fastVector<Pair> > xW; //W * X as defined in the assignment
        for (int j = 0;j < numKlass; j ++)
        {
            fastVector<Pair> fastweight (weights.col); //loads the sparse vector from a row of the weight matrix W.
            for (int idx = 0; idx < weights.col; idx ++)
                fastweight.vec[idx] = Pair(idx,weights.entries[j][idx]);
            
            //computes the product of w * X (both are sparse), then stores it in the variable "xW".
            fastVector<Pair> wiXi = fastweight * X ;
            xW.push_back (wiXi);
        }
        
        //O(nk) is the time complexity.
        //number of ell values is m (the number of training examples)
       
        /*
         Computation of probabilities
         */
        for (int ell = 0 ; ell < m; ell++)
        {
            int maxID = 0;
            /*
             Finds the class with highest value of w * X. We do it so that the exponential values are within numerical range.
             */
            for (int j = 0; j + 1 < numKlass; j++)
                if (xW[j].vec[ell].second > xW[maxID].vec[ell].second)
                    maxID = j;
            
            double denom = 1;
            //computes the denominator of probability equation below
            
            swap (xW[maxID].vec[ell], xW[0].vec[ell]);
            
            //denominator calculation for the first class here.
            for (int j = 1; j + 1 < numKlass; j ++)
                denom = denom + exp (xW[j].vec[ell].second - xW[0].vec[ell].second);
            
            //next divide the numerator on both top and bottom.
            double denom2 = 1.0/exp (xW[0].vec[ell].second);
            
            //probability of first class with row ell
            prob.entries[0][ell] = 1.0 / (denom + denom2);
            
            //take log
            denom = log (denom);
            
            for (int j = 1; j + 1 < numKlass; j ++)
            {
                //update the denominator
                denom = denom + (xW[j-1].vec[ell].second - xW[j].vec[ell].second);
                denom2 = 1.0 / (exp(xW[j].vec[ell].second));
                //probability of class j with row ell
                prob.entries[j][ell] = 1.0/ (exp(denom) + denom2);
            }
            
            //swap things back
            swap (xW[maxID].vec[ell], xW[0].vec[ell]);
            swap (prob.entries[maxID][ell], prob.entries[0][ell]);
          
            double sum = 0;
            for (int j = 0;j + 1< numKlass; j ++)
                sum = sum + prob.entries[j][ell];
            
            //substract the rest to get the last class.
            prob.entries[numKlass-1][ell] = abs(1 - sum);
        }
        //O(km) time complexity
        
        
        cout <<"Running iteration "<<iter<<endl;
        
        //gets accuracy and updates the bestweights.
        double accuracy = print_classified_accuracy (Y,weights,X, prob, classification, lambda, xW);
        if (accuracy > topaccuracy)
        {
            bestWeights = weights;
            topaccuracy = accuracy;
        }
    
        for (int j = 0;j < numKlass; j ++)
        {
            //updates weight with the penalty term.
            for (int idx = 0; idx < weights.col ; idx ++)
                weights.entries[j][idx] = weights.entries[j][idx] - eta * weights.entries[j][idx] * lambda;
    
            //runs over all rows to update weight values according to equation (29) in CMU notes.
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
        
        //some debugging output
        cout<<"Weights look like ";
               
        for (int j = 0 ; j < 5; j ++)
            cout<<weights.entries[0][j]<<" ";
                
        for (int j = 0 ; j < 5; j ++)
            cout<<weights.entries[0][weights.col - j - 1]<<" ";
                
        cout << endl;
    }
    
    //returns bestweights.
    
    return bestWeights ;
}

//debug code
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


//debug code
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

 referred this document at some point
 https://stackoverflow.com/questions/35419882/cost-function-in-logistic-regression-gives-nan-as-a-result
 
 */

//normalizes a sparseMatrix according to columns
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

//debug code
template<class Object>
void print_fastVector (fastVector<Object>& weights)
{
    for (int i = 0 ;i < weights.len; i ++)
        cout << weights.vec[i].second << " ";
    
    cout << endl;
    
    return ;
}


//debug code
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

//takes the input of labels and stores in labelids
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

//takes the input of words and stores in wordids

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

//runs the test data.

void run_test_data (int m, int k, int n, double eta, double lambda, Matrix<double> delta, sparseMatrix<Pair> X, vector<double> Y,
                       Matrix<double> weights, Matrix<double> prob)
{
    FILE *in = fopen ("testing.csv","r");
    
    char buffer[MAXSTR]; //FILE buffer
    vector<Pair> values ;
    vector<vector<Pair> > M;
    vector<int> docIds;
    vector<int> docClass;
    
    while (fgets (buffer, MAXSTR, in)){ //line by line input taken into buffer.
        char *token = strtok (buffer, ",");
        int idx = 0;
        while (token)
        {
            int num = atoi (token); //each number tokenized in the variable num
            values.push_back (Pair (idx ++ , num));
            
            token = strtok (NULL, ",");
        }
        
        vector<Pair> columnVec ;
        columnVec.push_back (Pair(0,1.0));
        minVal[0] = 0;
        docIds.push_back (values[0].second); //first value is document ID
        for (int j = 1;j < values.size (); j ++)
        {
            maxVAL[j] = max (maxVAL[j], values[j].second * 1.0); //column max
            minVal[j] = min (minVal[j], values[j].second * 1.0); //column min
          
            if (values[j].second)
                columnVec.push_back (Pair (j,values[j].second)); //adds the value of sparse vector if it is non-zero
        }
        
        M.push_back (columnVec); //adds the row to matrix.
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
    
    sparseMatrix<Pair> testX (M); //initializes test matrix X.
    
    fclose (in);

    int totalvalidentries = 0;
    
    for (int i = 0 ;i < m; i ++)
    {
  //      cout <<"Non zero valid entries in row "<<i<< " is "<< X.entries[i].size()<<endl;
        totalvalidentries = totalvalidentries + testX.entries[i].size ();
    }
    
    cout << "total number of valid entries is " << totalvalidentries << endl;

    //initialize_weights <Pair, double> (X,weights,Y);
    if (NORMALIZED) //normalizes if 1
        normalize (testX);
    
    
    //means the same as the gradient descent function except it runs on test matrix "TestX".
    int numKlass = weights.row ;
    vector<fastVector<Pair> > xW;
    for (int j = 0;j < numKlass; j ++)
    {
        fastVector<Pair> fastweight (weights.col);
        for (int idx = 0; idx < weights.col; idx ++)
            fastweight.vec[idx] = Pair(idx,weights.entries[j][idx]);
        
        fastVector<Pair> wiXi = fastweight * testX ;
        
        xW.push_back (wiXi);
    }
    
    //O(nk)
    //number of ell values is m (the number of training examples)
   
    //computes the probabilities with the weights found in training via same method inside gradient descent iterations.
    
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
    
    FILE *out = fopen("classification.csv","w");
    int countcorrect = 0;
    
    //classifies according to the highest probability.
    
    for (int ell = 0; ell < testX.row; ell ++)
    {
        int klass = 0;
        for (int j  = 0; j < k; j ++)
            if (prob.entries[j][ell] > prob.entries[klass][ell])
                klass = j ;
        
        docClass.push_back (klass);
        
      //  if (abs (klass - Y[ell]) < 1e-9)
        //    countcorrect ++;
        
        fprintf (out, "%d, %d\n", docIds[ell], klass);
    }
    
  //  cout <<"Percentage "<< countcorrect * 100.0/testX.row << endl;
    
    fclose (out);
    
    return ;
}


void print_confusion_matrix (vector<double> Y, Matrix<double> weights, sparseMatrix<Pair> X, Matrix<double> prob, double lambda, vector<double> docClass)
{
    FILE *out = fopen ("confusion_matrix.txt", "w");
    int numKlass = weights.row ;
    int countcorrect = 0;
    vector<vector<int> > C (20, vector<int> (20,0));
    
    for (int ell = 0; ell < X.row; ell ++)
    {
        int klass = 0;
        for (int j  = 0; j < numKlass; j ++)
            if (prob.entries[j][ell] > prob.entries[klass][ell])
                klass = j ;
        
        //finds the class of document ell and adds the counter if there is confusion.
        C[klass][docClass[ell]]++;
    }
    
    for (int i = 0;i < 20; i ++)
    {
        for (int j = 0; j < 20; j ++)
            fprintf (out, "%d ", C[i][j]);
        fprintf (out, "\n");
    }
    
    fclose (out);
    
    return;
}

void take_training_input_then_test ()
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
    
    while (fgets (buffer, MAXSTR, in)){ //line by line input taken into buffer.
        char *token = strtok (buffer, ",");
        int idx = 0;
        while (token)
        {
            int num = atoi (token); //each number tokenized in the variable num
            values.push_back (Pair (idx ++ , num));
            
            token = strtok (NULL, ",");
        }
        
        vector<Pair> columnVec ;
        columnVec.push_back (Pair(0,1.0));
        minVal[0] = 0;
        maxVAL[0] = maxVAL[1];
        docIds.push_back (values[0].second); //first value is document ID
        for (int j = 1;j + 1 < values.size (); j ++)
        {
            maxVAL[j] = max (maxVAL[j], values[j].second * 1.0); //column Max
            minVal[j] = min (minVal[j], values[j].second * 1.0); //column Min
          
            if (values[j].second)
                columnVec.push_back (Pair (j,values[j].second)); //non-zero values entered to sparse Matrix
        }
        
        docClass.push_back (values.back().second); //last value is the class of training document.
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
    //initialize data as below with notations defined in assignment
    Matrix<double> delta (k,m);
    sparseMatrix<Pair> X (M);
    vector<double> Y (docClass);
    Matrix<double> weights (k, n + 1);
    Matrix<double> probability (k,m);
    
    fclose (in);
    
    normal_distribution<double> gaussian (0.0, 1); //a normal distribution
    
    for (int i = 0;i < k; i ++)
        for (int j = 0;j <= n; j ++)
            weights.entries[i][j] = gaussian (generator); //weights initialized according the gaussian above.
    
    int totalvalidentries = 0;
    
    for (int i = 0 ;i < m; i ++)
    {
  //      cout <<"Non zero valid entries in row "<<i<< " is "<< X.entries[i].size()<<endl;
        totalvalidentries = totalvalidentries + X.entries[i].size ();
    }
    
    cout << "total number of valid entries is " << totalvalidentries << endl;

    //initialize delta.
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
    
    
    Matrix<double> bestweights =  gradient_descent (m, k, n, eta, lambda, delta, X, Y, weights, probability);
    run_test_data (m, k, n, eta, lambda, delta, X, Y, bestweights, probability);
    print_confusion_matrix (Y,weights,X, probability, lambda, docClass);
    
    return ;
}



int main()
{
    test_code_matrix ();
    test_code_sparseMatrix ();
    
    take_word_input ();
    take_label_input ();
    take_training_input_then_test ();

    return 0;
}
