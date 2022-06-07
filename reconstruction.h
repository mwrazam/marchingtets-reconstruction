#pragma once

#include <vector>
#include <stack>
#include <unordered_map>
#include <iostream>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cassert>
#include <fstream>
#include <string>

#include "kdtree.hpp"

namespace mtr {
    using namespace std;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // comment : why is switching between vector and matrix required?
    template <typename T, int cols> 
    void matrix_to_2dvector(
        Eigen::Matrix<T, -1, cols> const& matrix, // the matrix to convert
        vector<vector<T>> &vec // the vector to fill
    )
    {
        for (int i = 0; i < matrix.rows(); i++) 
        {
            vec.emplace_back(vector<T>{matrix(i,0), matrix(i,1), matrix(i,2)});
        }
    }

    // TODO: move utility functions into the utilities file
    template <typename T, int cols>
    void vector2d_to_matrix(
        vector<vector<T>> const& vec, // input 2d vector
        Eigen::Matrix<T, -1, cols> &M // matrix to fill
    )
    {
        M = Eigen::Matrix<T, -1, cols>::Zero(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++)
        {
            for (int j = 0; j < vec[i].size(); j++)
            {
                M(i, j) = vec[i][j];
            }
        }
    }

    template <typename T, int cols>
    void extract_rows(
        const Eigen::Matrix<T, -1, cols> &M, // matrix to extract rows from
        const vector<int> &indices, // indicies to extract
        Eigen::Matrix<T, -1, cols> &M2 // matrix to extract rows into
    )
    {
        M2 = Eigen::Matrix<T, -1, cols>::Zero(indices.size(), cols);
        // TODO: Is there a way to do this using matrix indices instead of using for loop??
        for (int i = 0; i < indices.size(); i++)
        {
            M2.row(i) << M.row(indices[i]);
        }
    }

    // suggestion : try to implement it more efficiently
    // suggestion : if vals is sorted, then std::find can be replaced with std::binary_search
    template <typename T>
    void replace_values(
        Eigen::Matrix<T, -1, 3> &M, // matrix to update
        const vector<T> &vals, // values to replace
        T new_val // new value
    )
    { // could improve this by using a hash map instead, i.e. f_premerge => f_a // nima : I disagree
        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.cols(); j++)
            {
                // check if this value is contained in our vals
                if(find(vals.begin(), vals.end(), M(i,j)) != vals.end() )
                {
                    M(i,j) = new_val;
                }
            }
        }
    }

    // suggestion : remove it if it's not being used
    // suggestion : nn_search can be implemented using a Kdtree
    // suggestion : embarassignly parallel
    template <typename T>
    void pca_normals(
        const Eigen::Matrix<T, -1, 3> &V, // vertices
        Eigen::Matrix<T, -1, 3> &N, // normals to fill
        int k // number of neighbours to use for plane fit
    )
    {
        N = Eigen::Matrix<T, -1, 3>::Zero(V.rows(), V.cols()); // 1 normal per vertex
        Eigen::Matrix<T, 1, 3> p;
        for (int i = 0; i < V.rows(); i++)
        {
            
            p = V.row(i);
            // 1. collect k nearest neighbours for this point
            pair<vector<int>, vector<T>> nn = nearest_neighbours(V, p, k);
            Eigen::Matrix<T, -1, 3> P;
            extract_rows<T, 3>(V, nn.first, P);

            // 2. Subtract centroid from each neighor point
            Eigen::Matrix<T, 1, 3> m = P.colwise().mean();
            P = P.rowwise() - m;
            
            // 3. Compute Scatter matrix
            Eigen::Matrix<T, -1, -1> S = P.transpose() * P;
            
            // 4. Compute eigenvalues of S
            Eigen::EigenSolver<Eigen::Matrix<T, -1, -1>> es(S);
            Eigen::Matrix<T, 3, 3> eival_matrix = es.pseudoEigenvalueMatrix();
            Eigen::Matrix<T, 3, 1> eigenvalues {eival_matrix(0,0), eival_matrix(1,1), eival_matrix(2,2)};
            Eigen::Matrix<T, 3, 3> eigenvectors = es.pseudoEigenvectors();

            // 5. Take the eigenvector corresponding to smallest eigenvalue as normal
            int min_idx = 0;
            T current_max = eival_matrix(min_idx, min_idx);
            for (int j = min_idx + 1; j < 3; j++)
            {
                if (eival_matrix(j,j) < current_max)
                {
                    min_idx = j;
                    current_max = eival_matrix(j, j);
                }
            }
            
            // 6. Set the normal
            N.row(i) = eigenvectors.row(min_idx);
        }
    }

    // suggestion : embarassignly parallel
    template <typename T>
    pair<Eigen::Matrix<T, 1, 3>, T> compute_constraint_and_value(
        const Eigen::Matrix<T, 1, 3> &v, // given point
        const Eigen::Matrix<T, 1, 3> &n, // normal for given point
        T eps, // distance factor
        T max_dist // maximum allowed distance from given point to new constraint point
    )
    {
        Eigen::Matrix<T, 1, 3> c;
        T d;
        while (true)
        { // compute a new constraint point
            c = v + n * eps;
            T dist = (c - v).norm();
            if (dist < max_dist)
            { // if the point is within allowed distance keep it
                d = (n * eps).norm();
                break; // success
            }
            else
            { // otherwise reduce eps and try again
                eps /= 2; 
            }
        }
        return make_pair(c, d);
    }

    template <typename T>
    T distance(Kdtree::CoordPoint const& a, Kdtree::CoordPoint const& b){
        size_t n = a.size();
        T result = 0.0;
        for (size_t i=0; i<n; i++)
            result += pow(a[i]-b[i], 2);
        return sqrt(result);
    }

    // suggestion : use a kdtree to find nearest neighbors
    // suggestion : assign tasks to differnet threads
    // suggestion : seems like a great fit for SIMD (inside the for loop)
    template <typename T>
    void generate_constraints_and_values(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        const Eigen::Matrix<T, -1, 3> &N, // original normals
        Eigen::Matrix<T, -1, 3> &C, // constraint points to fill
        Eigen::Matrix<T, -1, 1> &D, // constraint values to fill
        T eps // distance control between original and constraint points
    )
    {
        C = Eigen::Matrix<T, -1, 3>::Zero(V.rows()*3, V.cols()); // 3 constraints per point
        D = Eigen::Matrix<T, -1, 1>::Zero(V.rows()*3, 1); // 1 constraint value per constraint point

        // KDTree construction
        Kdtree::KdNodeVector nodes;
        for (int i = 0; i < V.rows(); i++){
            Kdtree::CoordPoint point(3);
            point = {V(i, 0), V(i, 1), V(i, 2)};
            nodes.push_back(Kdtree::KdNode(point));
        }
        Kdtree::KdTree tree(&nodes);

        for (int i = 0; i < V.rows(); i++)
        {
            Eigen::Matrix<T, 1, 3> p = V.row(i);
            Eigen::Matrix<T, 1, 3> n1 = N.row(i);
            Eigen::Matrix<T, 1, 3> n2 = -N.row(i);

            //kNN search
            Kdtree::KdNodeVector nn;
            Kdtree::CoordPoint query_point(3);
            query_point = {V(i, 0), V(i, 1), V(i, 2)};
            tree.k_nearest_neighbors(query_point, 2, &nn);
            // the second nearest is desired since the nearest is the same point
            T nn_dist = distance<T>(query_point, nn[1].point);


            // first set of constraints and points are the vertices themselves
            C.row(i) = V.row(i); // use data vertices as one set of constraints
            D(i) = 0.0; // implict function evaluates to 0 at the vertex

            // second set of constraints, points outside of the mesh
            pair<Eigen::Matrix<T, 1, 3>, T> c_d1 = compute_constraint_and_value(p, n1, eps, nn_dist);
            C.row(V.rows() + i) = c_d1.first;
            D(V.rows() + i) = c_d1.second;

            // third set of constraints, points inside the mesh
            pair<Eigen::Matrix<T, 1, 3>, T> c_d2 = compute_constraint_and_value(p, n2, eps, nn_dist);
            C.row(V.rows()*2 + i) = c_d2.first;
            D(V.rows()*2 + i) = -c_d2.second;
        }
        
    }
 
    // suggestion : ask for clarification about this function
    template <typename T>
    void generate_grid(
        Eigen::Matrix<T, -1, 3> &V, // vertices of grid to fill
        const Eigen::Matrix<int, 1, 3> &nt, // number of grid points in x, y, z directions
        const Eigen::Matrix<T, 1, 3> &gmin, // grid minimum point
        const Eigen::Matrix<T, 1, 3> &gmax, // grid maximum point
        const Eigen::Matrix<T, 1, 3> &pad // additional padding to apply to grid
    )
    {
        int nx = nt(0), ny = nt(1), nz = nt(2);
        V = Eigen::Matrix<T, -1, 3>::Zero(nx*ny*nz, 3);

        // Compute points on each coordinate
        Eigen::ArrayXf x_points = Eigen::ArrayXf::LinSpaced(nx, gmin(0) - pad(0), gmax(0) + pad(0));
        Eigen::ArrayXf y_points = Eigen::ArrayXf::LinSpaced(nx, gmin(1) - pad(1), gmax(1) + pad(1));
        Eigen::ArrayXf z_points = Eigen::ArrayXf::LinSpaced(nx, gmin(2) - pad(2), gmax(2) + pad(2));
        // Generate vertices of tet grid
        int vi = 0;
        for (int k = 0; k < z_points.rows(); k++)
        {
            for (int j = 0; j < y_points.rows(); j++)
            {
                for (int i = 0; i < x_points.rows(); i++)
                {
                    V.row(vi) = Eigen::RowVector3d(x_points(i), y_points(j), z_points(k));
                    vi++;
                }
            }
        }
    }
    
    // suggestion : seems like race condition may happen if parallelized!
    template <typename T>
    void generate_tets_from_grid(
        Eigen::Matrix<int, -1, 4> &TF, // tet definitions
        const Eigen::Matrix<T, -1, 3> &TV, // grid vertices
        const Eigen::Matrix<int, 1, 3> &nt // number of grid points in x, y, z directions
    )
    {
        int nx = nt(0), ny = nt(1), nz = nt(2);
        // each 'cube' will have 6 tets in it, and each tet is defined by 4 vertices
        TF = Eigen::Matrix<int, -1, 4>::Zero((nx-1) * (ny-1) * (nz-1) * 6, 4);

        // keep track of which 'set' of tets we are on
        int ti = 0;

        // suggestion : good for SIMD
        // we will iterate over all 'cubes' in the grid and create 6 new tets
        for (int i = 0; i < nx - 1; i++)
        {
            for (int j = 0; j < ny - 1; j++)
            {
                for (int k = 0; k < nz - 1; k++)
                {
                    // collect the indices of the vertices of this cube
                    int v1 = (i+0)+nx*((j+0)+ny*(k+0));
                    int v2 = (i+0)+nx*((j+1)+ny*(k+0));
                    int v3 = (i+1)+nx*((j+0)+ny*(k+0));
                    int v4 = (i+1)+nx*((j+1)+ny*(k+0));
                    int v5 = (i+0)+nx*((j+0)+ny*(k+1));
                    int v6 = (i+0)+nx*((j+1)+ny*(k+1));
                    int v7 = (i+1)+nx*((j+0)+ny*(k+1));
                    int v8 = (i+1)+nx*((j+1)+ny*(k+1));

                    // form 6 tets using these vertices
                    TF.row(ti) << v1,v3,v8,v7;
                    TF.row(ti + 1) << v1,v8,v5,v7;
                    TF.row(ti + 2) << v1,v3,v4,v8;
                    TF.row(ti + 3) << v1,v4,v2,v8;
                    TF.row(ti + 4) << v1,v6,v5,v8;
                    TF.row(ti + 5) << v1,v2,v6,v8;
                    ti += 6; // increment our counter
                }
            }
        }
    }
    
    // suggestion : not computationally intensive but embarassignlt parallel
    template <typename T>
    void polynomial_basis_vector(
        const int degree, // degree of polynomial
        const Eigen::Matrix<T, 1, 3> &p, // point to compute basis vector for
        Eigen::Matrix<T, 1, -1> &b // basis vector to create
    )
    {
        T x = p(0), y = p(1), z = p(2);
        Eigen::RowVectorXd basis;

        if (degree == 0)
        {
            b = Eigen::Matrix<T, 1, -1>(1);
            b << 1;
        }

        if (degree == 1)
        {
            b = Eigen::Matrix<T, 1, -1>(4);
            b << 1, x, y, z;
        }

        if (degree == 2)
        {
            b = Eigen::Matrix<T, 1, -1>(10);
            b << 1, x, y, z, x*y, x*z, y*z, x*x, y*y, z*z;
        }
    }

    // suggestion : good fit for SIMD
    template <typename T>
    void generate_weights_matrix(
        const Eigen::Matrix<T, -1, 3> &P, // points for computation
        const Eigen::Matrix<T, 1, 3> &X, // ???what is this again????
        const double h, // height of welland function
        Eigen::Matrix<T, -1, -1> &W // weights matrix to build
    )
    {
        // Welland weights matrix is a diagonal square matrix matching the size of input points
        W = Eigen::Matrix<T, -1, -1>::Zero(P.rows(), P.rows());
        for (int i = 0; i < P.rows(); i++)
        {
            double r = (P.row(i) - X).norm();
            double w = (4 * r/h + 1)*(1 - r/h);
            w = pow(w, 4);
            W(i, i) = w;
        }
    }

    // suggestion : for-loop writes to independent elements - easy to parallelize
    template <typename T>
    void generate_basis_matrix(
        const Eigen::Matrix<T, -1, 3> &P, // points to use for generation
        Eigen::Matrix<T, -1, -1> &B // basis matrix to generate
    )
    {
        // compute basis vector for first point to determine basis matrix size
        Eigen::Matrix<T, 1, -1> bv;
        Eigen::Matrix<T, 1, 3> p {P.row(0)}; // suggestion : isn't it redundant
        polynomial_basis_vector<T>(2, p, bv); // suggestion : isn't it redundant

        B = Eigen::Matrix<T, -1, -1>::Zero(P.rows(), bv.size());
        for (int i = 0; i < P.rows(); i++)
        {
            p = P.row(i);
            polynomial_basis_vector(2, p, bv);
            B.row(i) = bv;
        }
    }

    // suggestion : once the kdtree is built, the rest can be done in parallel
    template <typename T>
    void compute_implicit_function_values(
        Eigen::Matrix<T, -1, 1> &fx, // implicit function values to compute
        const Eigen::Matrix<T, -1, 3> &TV, // vertices of grid
        const Eigen::Matrix<T, -1, 3> &C, // constraint points
        const Eigen::Matrix<T, -1, 1> &D, // values at constraint points
        double w // welland radius
    )
    {
        fx = Eigen::Matrix<T, -1, 1>::Zero(TV.rows()); // one function value per grid point
        Eigen::Matrix<T, 1, 3> p {TV.row(0)};
        Eigen::Matrix<T, 1, -1> b;
        polynomial_basis_vector<T>(2, p, b);
        int min_num_pts = b.size();

        // suggestion : construct a kd-tree and perform range-search - done!
        bool unique = false; // it doesn't matter if a node is already visited
        // constructing KDTree
        Kdtree::KdNodeVector nodes;
        for (int i = 0; i < C.rows(); i++)
        {
            Kdtree::CoordPoint point(3);
            point = {C(i, 0), C(i, 1), C(i, 2)};
            Kdtree::KdNode tmp(point);
            tmp.idx = i;
            nodes.push_back(tmp);
        }
        Kdtree::KdTree tree(&nodes);

        // evaluate the implict function at each point in the tet grid
        for (int i = 0; i < TV.rows(); i++)
        {
            Kdtree::CoordPoint point = {TV(i, 0), TV(i, 1), TV(i, 2)};
            Kdtree::KdNodeVector result;
            tree.range_nearest_neighbors(point, w, &result, unique);

            p = TV.row(i);
            // vector<int> pi = points_within_radius<T>(C, p, w);            

            // assume this point is outside of the mesh
            if(result.size() == 0)
            {
                fx(i) = 100.0;
            }
            else
            {
                vector<int> pi2(result.size());
                std::transform(result.begin(), result.end(), pi2.begin(), [](auto const& node){return node.idx;});
                // assert (pi.size() == pi2.size());
                // std::sort(pi.begin(), pi.end());
                // std::sort(pi2.begin(), pi2.end());
                // for (int idx = 0; idx < pi.size(); idx++)
                //     assert(pi[idx] == pi2[idx]);
                Eigen::Matrix<T, -1, 3> P;
                Eigen::Matrix<T, -1, 1> values;
                extract_rows<T, 3>(C, pi2, P);
                extract_rows<T, 1>(D, pi2, values);

                Eigen::Matrix<T, -1, -1> W;
                Eigen::Matrix<T, -1, -1> B;
                generate_weights_matrix<T>(P, p, w, W);
                generate_basis_matrix<T>(P, B);

                // Solve: (B.T*W*B)a = (B.T*W*D) 
                Eigen::Matrix<T, -1, -1> imd = B.transpose() * W;
                Eigen::Matrix<T, -1, -1> L;
                L = imd * B;
                Eigen::Matrix<T, -1, -1> R = imd * values;
                Eigen::Matrix<T, -1, 1> a = L.llt().solve(R);
                a = L.ldlt().solve(R);

                // Calculate function value
                Eigen::Matrix<T, 1, -1> gi;
                polynomial_basis_vector<T>(2, p, gi);
                T v = gi.dot(a);
                fx(i) = v;
            } 
        }
    }
    
    // suggestion : light computation required - embarassingly parallel
    template <typename T>
    Eigen::Matrix<T, 1, 3> generate_triangle_point(
        const Eigen::Matrix<T, 1, 3> &p1,  // point 1 of edge
        const Eigen::Matrix<T, 1, 3> &p2, // point 2 of edge
        const T v1, // value at point 1
        const T v2, // value at point 2
        T snap = 0.01 // threshold to snap vertices when close to grid edge point
    )
    {
        if (abs(v1) < snap)
        {
            return p1;
        }
        else if (abs(v2) < snap)
        {
            return p2;
        }
        else if (abs(v2-v1) < snap)
        {
            return p1;
        }
        else
        {
            double mu = v2 / (v2 - v1);
            return (mu * p1) + ((1-mu) * p2);
        }
    }
    
    // suggestion : v_i and f_i are dynamically allocated. if they are initially of size 9*|Tets| 
    // then the process can be done in parallel. However, some unused memory will be consumed.
    template <typename T>
    void marching_tetrahedra(
        const Eigen::Matrix<T, -1, 3> &G, // Tet grid vertices
        const Eigen::Matrix<int, -1, 4> &Tets, // Tets of the tet grid
        const Eigen::Matrix<T, -1, 1> &fx, // implict function values at each grid vertex
        Eigen::Matrix<T, -1, 3> &SV, // reconstructed mesh vertices
        Eigen::Matrix<int, -1, 3> &SF // reconstructed mesh faces
    )
    {

        vector<T> v_i; // holder for the mesh vertices we will build
        vector<int> f_i; // holder for the face definition
        int t = 0; // tri counter
        for (int i = 0; i < Tets.rows(); i++)
        {
            // march over each tet in our set of tets
            int config = 0; // the 'configuration' we have for the current tet

            // current tet
            Eigen::Matrix<int, 1, 4> tet = Tets.row(i);
            
            // vertices of current tet
            Eigen::Matrix<T, 1, 3> p0 = G.row(tet(0));
            Eigen::Matrix<T, 1, 3> p1 = G.row(tet(1));
            Eigen::Matrix<T, 1, 3> p2 = G.row(tet(2));
            Eigen::Matrix<T, 1, 3> p3 = G.row(tet(3));

            // implicit function values at current tet
            T v0 = fx(tet(0));
            T v1 = fx(tet(1));
            T v2 = fx(tet(2));
            T v3 = fx(tet(3));

            // suggestion - make it branchless as follows
            // Determine which type of configuration we have
            if (v0 < 0.0) {config += 1;}
            if (v1 < 0.0) {config += 2;}
            if (v2 < 0.0) {config += 4;}
            if (v3 < 0.0) {config += 8;}

            // config += 1*(v0 < 0.0) + 2*(v1 < 0.0) + 4*(v2 < 0.0) + 8*(v3 < 0.0);
            

            // Create tris based on the configuration we have
            switch (config)
            {
                case 0: case 15: // 0000 or 1111
                {
                    break;
                }
                case 1: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 2:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p1, p0, v1, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 3: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p2, v1, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = t1a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point<T>(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 4: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p2, p0, v2, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p2, p1, v2, v1);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 5: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point<T>(p2, p3, v2, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 6: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = t2a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 7:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p3, p0, v3, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p3, p2, v3, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 8:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p3, p0, v3, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p3, p2, v3, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 9:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = t2a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 10:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point<T>(p2, p3, v2, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 11:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p2, p0, v2, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p2, p1, v2, v1);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 12:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p2, v1, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t1a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 13:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p1, p0, v1, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 14:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
    
        // Note: Using conservative matrix resizing is super slow, so we
        // simply create and fill a matrix using interm vectors

        // maybe convert the above and following sections to be 2d vectors and use conversion
        // function instead for cleaner code
        SV = Eigen::Matrix<T, -1, 3>::Zero(v_i.size()/3, 3);
        SF = Eigen::Matrix<int, -1, 3>::Zero(f_i.size()/3,3);
        for(int i = 0; i < v_i.size(); i = i + 3)
        {
            SV.row(i/3) << v_i[i], v_i[i+1], v_i[i+2];
        }
        for (int i = 0; i < f_i.size(); i = i + 3)
        {
            SF.row(i/3) << f_i[i], f_i[i+1], f_i[i+2];
        }
    }

    // suggestion : not paralleizable!!! range_search requires unique results and it means that
    // iterations are dependent. Moreover, replace_values may cause race condition if merge_vertices 
    // is parallelized
    template <typename T>
    void merge_vertices(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        Eigen::Matrix<int, -1, 3> &F, // faces
        T eps, // merge threshold
        Eigen::Matrix<T, -1, 3> &V2 // new vertices
    )
    {

        // constructing KDTree
        Kdtree::KdNodeVector nodes;
        for (int i = 0; i < V.rows(); i++)
        {
            Kdtree::CoordPoint point(3);
            point = {V(i, 0), V(i, 1), V(i, 2)};
            Kdtree::KdNode tmp(point);
            tmp.idx = i;
            nodes.push_back(tmp);
        }

        Kdtree::KdTree tree(&nodes);

        vector<Eigen::Matrix<T, 1, 3>> new_verts; // Eigen conservative resize is too heavy, this should be faster
        int current_new_v_i = 0;

        bool unique =  true; // already visited points are not of our interest

        for (int i=0; i<V.rows(); i++){
            vector<int> to_merge;
            // range search on KDTree
            Kdtree::CoordPoint point = {V(i, 0), V(i, 1), V(i, 2)};
            Kdtree::KdNodeVector result;
            tree.range_nearest_neighbors(point, eps, &result, unique);
            if (result.size() > 0){ // if point was not visited
                Eigen::Matrix<T, 1, 3> P {0.0, 0.0, 0.0}; // new point
                for (int j = 0; j < result.size(); j++)
                {
                    to_merge.push_back(result.at(j).idx);
                    Eigen::Matrix<T, 1, 3> n_P;
                    n_P << result.at(j).point.at(0), result.at(j).point.at(1), result.at(j).point.at(2);
                    P += n_P;
                }
                P /= to_merge.size();
                new_verts.emplace_back(P);
                // suggestion : can't we accumulate them and call the replace_values once?
                replace_values(F, to_merge, current_new_v_i);
                current_new_v_i++;
            }
        }

        // fill V2 with new verts
        V2 = Eigen::Matrix<T, -1, 3>::Zero(new_verts.size(), V.cols());
        for (int i = 0; i < new_verts.size(); i++)
        {
            V2.row(i) = new_verts[i];
        }
    }

    template <typename T>
    void depth_first_search(
        const unordered_map<T, vector<T> > &adj, // adjacency list to search
        vector<bool> &visisted, // set of visited nodes
        vector<T> &output, // build output path through the graph
        T u // element to search for
    )
	{
        // TODO: Move this over into a graph class
		visisted[u] = true;
		output.push_back(u);

		for (T i = 0; i < adj.at(u).size(); i++)
		{
			T uu = adj.at(u)[i]; // suggestion : make a local copy of adj.at(u) instead of accessing it multiple times
			if (!visisted[uu])
			{
				depth_first_search(adj, visisted, output, uu);
			}
		}
	}

    // suggestion : make sure that vec is sorted, then perform binary search
    // TODO: move the graph related functions into a graph class
    template <typename T>
    void add_edge(
        vector<T> &vec, // connect from all of these nodes
        T dst // to this node
    )
	{
        
		if (find(vec.begin(), vec.end(), dst) == vec.end())
		{
			vec.push_back(dst);
		}
	}

    // suggestion : replace unordered_map with vector<vector<T>> so that dfs will execute faster as well!
    template <typename T> // this is probably useless, should enforce int type, unless we want to allow for size_t
	unordered_map<T, vector<T>> adjacency_list( // should be adjacency_list_from_faces or something
        const Eigen::Matrix<T, -1, 3> &M // input faces
    )
	{
        
		unordered_map<T, vector<T> > adj;
		for (int i = 0; i < M.rows(); i++)
		{
			int v1 = M(i, 0);
			int v2 = M(i, 1);
			int v3 = M(i, 2);

            // suggestion : at least, use the pointer returned by find to prevent using hash_map twice
			if (adj.find(v1) == adj.end()) { adj.insert({ v1, vector<T>() }); }
			if (adj.find(v2) == adj.end()) { adj.insert({ v2, vector<T>() }); }
			if (adj.find(v3) == adj.end()) { adj.insert({ v3, vector<T>() }); }

			add_edge(adj[v1], v2);
			add_edge(adj[v1], v3);
			add_edge(adj[v2], v1);
			add_edge(adj[v2], v3);
			add_edge(adj[v3], v1);
			add_edge(adj[v3], v2);
		}
		return adj;
	}

    // suggestion : not easy to parallelize! seen vector makes iterations dependent on each other.
    template <typename T>
	unordered_map<int, vector<int>> connected_components(
        const Eigen::Matrix<T, -1, 3> &V, // vertices
        const Eigen::Matrix<int, -1, 3> &F // face definitions used to compute edges
    ) 
	{
        // suggestion : try to get rid of unordered_maps
		unordered_map<int, vector<int> > adj = adjacency_list<int>(F); // construct adjacency list of connected edges
        unordered_map<int, vector<int>> cc;
		vector<bool> seen(V.rows(), false); // hold which items we have seen so far

		// perform depth first search on each key in the adj_list
        int idx = 0;
		for (auto kv : adj)
		{
			vector<int> output;
			depth_first_search(adj, seen, output, kv.first);
			if (output.size() > 1) // only interested on connected components with more than one vertex
			{
                cc.insert({idx, output});
                idx++;
			}
		}
        return cc;
	}

    // suggestion : new_F seems to be redundant. Why not appending to F2?
    // suggestion : if some fixed memory is allocated to F2, it can be parallelized! 
    template <typename T>
    void largest_connected_component(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        const Eigen::Matrix<int, -1, 3> &F, // original faces 
        Eigen::Matrix<int, -1, 3> &F2 // faces defining largest connected component
    )
    {
        // suggestion : try to replace unordered_map with vector
        unordered_map<int, vector<int>> cc = connected_components<T>(V, F);
        int largest_cc_size = 0;
        int largest_cc_key;
        for (auto kv : cc)
        {
            if(kv.second.size() > largest_cc_size)
            {
                largest_cc_key = kv.first;
                largest_cc_size = kv.second.size();
            }
        }
        
        vector<bool> keep(F.rows(), false); // store the faces we want to keep for quick lookup
        vector<Eigen::Matrix<int, 1, 3>> new_F; // temporary holder for the faces we want to keep

        // flag the faces we want to keep
        for (int f : cc[largest_cc_key])
        {
            keep[f] = true;
        }

        for (int i = 0; i < F.rows(); i++)
        { // note: to keep a face, all three vertices must show up in our keep map
            if(keep[F(i,0)] && keep[F(i,1)] && keep[F(i,2)])
            {
                new_F.emplace_back(F.row(i));
            }
        }

        F2 = Eigen::Matrix<int, -1, 3>::Zero(new_F.size(), 3);
        for (int i = 0; i < new_F.size(); i++)
        {
            F2.row(i) = new_F[i];
        }
    }

    template <typename T>
    pair<vector<vector<T>>, vector<vector<int>>> reconstruction(
        const vector<vector<T>> &vertices, // input data vertices
        const vector<vector<T>> &normals, // input data normals
        const int nx, // reconstruction grid resolution x-direction
        const int ny, // reconstruction grid resolution y-direction
        const int nz, // reconstruction grid resolution z-direction
        const double mesh_scale, // scale original mesh
        const double welland_radius, // welland function height parameter
        const double epsilon // distance control parameter
    ) 
    {
        Eigen::Matrix<T, -1, 3> V; // input vertices
        Eigen::Matrix<T, -1, 3> N; // input normals

        vector2d_to_matrix<T, 3>(vertices, V);
        vector2d_to_matrix<T, 3>(normals, N);

        auto t1 = high_resolution_clock::now();
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double;

        // PCA normals don't well until we find a way to correct inversions
        /*
        cout << "PCA normals... ";
        t1 = high_resolution_clock::now();
        pca_normals(V, N, 10);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;
        */
        
        V = V * mesh_scale; // welland weights computation performs better with scaling
        V =  V.rowwise() - V.colwise().mean(); // re-center data on origin

        // ### Step 1: Compute constraint points and values ###
        Eigen::Matrix<T, -1, 3> C; // implict function constraint points
        Eigen::Matrix<T, -1, 1> D; // implict function values at corresponding constraints
        double eps = epsilon * (V.colwise().minCoeff() - V.colwise().maxCoeff()).norm();
        t1 = high_resolution_clock::now();
        cout << "Constraints and values... ";
        generate_constraints_and_values<T>(V, N, C, D, eps);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;

        // ### Step 2: Generate Tet grid ###
        Eigen::Matrix<T, -1, 3> TV; //vertices of tet grid
        Eigen::Matrix<int, -1, 4> TF; // tets of tet grid
        Eigen::Matrix<int, 1, 3> num_tets {nx, ny, nz}; 
        Eigen::Matrix<T, 1, 3> gmin = V.colwise().minCoeff(); // grid minimum point
        Eigen::Matrix<T, 1, 3> gmax = V.colwise().maxCoeff(); // grid maximum point
        Eigen::Matrix<T, 1, 3> padding {eps, eps, eps}; // additional padding for the grid
        t1 = high_resolution_clock::now();
        cout << "Tet grid... ";
        generate_grid<T>(TV, num_tets, gmin, gmax, padding); // generate grid
        generate_tets_from_grid<T>(TF, TV, num_tets);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;

        // ### Step 3: Compute implict function values at all tet grid points ###
        Eigen::Matrix<T, -1, 1> fx;
        t1 = high_resolution_clock::now();
        cout << "Implict function values... ";
        compute_implicit_function_values<T>(fx, TV, C, D, welland_radius);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;

        // ### Step 4: March tets and extract iso surface ###
        Eigen::Matrix<T, -1, 3> SV; // vertices of reconstructed mesh
        Eigen::Matrix<int, -1, 3> SF; // faces of reconstructed mesh
        t1 = high_resolution_clock::now();
        cout << "Marching tetrahedra... ";
        marching_tetrahedra<T>(TV, TF, fx, SV, SF);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;

        // ### Cleanup ###
        Eigen::Matrix<T, -1, 3> SV2;
        t1 = high_resolution_clock::now();
        cout << "Vertex merge... ";
        merge_vertices<T>(SV, SF, 0.00001, SV2);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;

        Eigen::Matrix<int, -1, 3> SF2;
        t1 = high_resolution_clock::now();
        cout << "Largest connected component search... ";
        largest_connected_component<T>(SV2, SF, SF2);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;

        // convert data back to vector format
        vector<vector<T>> V2; // reconstructed vertices
        vector<vector<int>> F2; // reconstructed faces
        matrix_to_2dvector<T, 3>(SV2, V2);
        matrix_to_2dvector<int, 3>(SF2, F2);

        return pair<vector<vector<T>>, vector<vector<int>>>(V2, F2);
    }


}

namespace vectorization {
    using namespace std;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // comment : why is switching between vector and matrix required?
    template <typename T, int cols> 
    void matrix_to_2dvector(
        Eigen::Matrix<T, -1, cols> const& matrix, // the matrix to convert
        vector<vector<T>> &vec // the vector to fill
    )
    {
        for (int i = 0; i < matrix.rows(); i++) 
        {
            vec.emplace_back(vector<T>{matrix(i,0), matrix(i,1), matrix(i,2)});
        }
    }

    // TODO: move utility functions into the utilities file
    template <typename T, int cols>
    void vector2d_to_matrix(
        vector<vector<T>> const& vec, // input 2d vector
        Eigen::Matrix<T, -1, cols> &M // matrix to fill
    )
    {
        M = Eigen::Matrix<T, -1, cols>::Zero(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++)
        {
            for (int j = 0; j < vec[i].size(); j++)
            {
                M(i, j) = vec[i][j];
            }
        }
    }
    
    template <typename T, int cols>
    void extract_rows(
        const Eigen::Matrix<T, -1, cols> &M, // matrix to extract rows from
        const vector<int> &indices, // indicies to extract
        Eigen::Matrix<T, -1, cols> &M2 // matrix to extract rows into
    )
    {
        M2 = Eigen::Matrix<T, -1, cols>::Zero(indices.size(), cols);
        // TODO: Is there a way to do this using matrix indices instead of using for loop??
        for (int i = 0; i < indices.size(); i++)
        {
            M2.row(i) << M.row(indices[i]);
        }
    }

    // suggestion : try to implement it more efficiently
    // suggestion : if vals is sorted, then std::find can be replaced with std::binary_search
    template <typename T>
    void replace_values(
        Eigen::Matrix<T, -1, 3> &M, // matrix to update
        const vector<T> &vals, // values to replace 
        T new_val // new value
    )
    { // could improve this by using a hash map instead, i.e. f_premerge => f_a

        int cnt = vals.size();

        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.cols(); j++)
            {
                // check if this value is contained in our vals
                if(find(vals.begin(), vals.end(), M(i,j)) != vals.end() )
                {
                    M(i,j) = new_val;
                    cnt--;
                    if (cnt == 0)   return;
                }
            }
        }
    }

    // suggestion : remove it if it's not being used
    // suggestion : nn_search can be implemented using a Kdtree
    // suggestion : embarassignly parallel
    template <typename T>
    void pca_normals(
        const Eigen::Matrix<T, -1, 3> &V, // vertices
        Eigen::Matrix<T, -1, 3> &N, // normals to fill
        int k // number of neighbours to use for plane fit
    )
    {
        N = Eigen::Matrix<T, -1, 3>::Zero(V.rows(), V.cols()); // 1 normal per vertex
        Eigen::Matrix<T, 1, 3> p;
        for (int i = 0; i < V.rows(); i++)
        {
            
            p = V.row(i);
            // 1. collect k nearest neighbours for this point
            pair<vector<int>, vector<T>> nn = nearest_neighbours(V, p, k);
            Eigen::Matrix<T, -1, 3> P;
            extract_rows<T, 3>(V, nn.first, P);

            // 2. Subtract centroid from each neighor point
            Eigen::Matrix<T, 1, 3> m = P.colwise().mean();
            P = P.rowwise() - m;
            
            // 3. Compute Scatter matrix
            Eigen::Matrix<T, -1, -1> S = P.transpose() * P;
            
            // 4. Compute eigenvalues of S
            Eigen::EigenSolver<Eigen::Matrix<T, -1, -1>> es(S);
            Eigen::Matrix<T, 3, 3> eival_matrix = es.pseudoEigenvalueMatrix();
            Eigen::Matrix<T, 3, 1> eigenvalues {eival_matrix(0,0), eival_matrix(1,1), eival_matrix(2,2)};
            Eigen::Matrix<T, 3, 3> eigenvectors = es.pseudoEigenvectors();

            // 5. Take the eigenvector corresponding to smallest eigenvalue as normal
            int min_idx = 0;
            T current_max = eival_matrix(min_idx, min_idx);
            for (int j = min_idx + 1; j < 3; j++)
            {
                if (eival_matrix(j,j) < current_max)
                {
                    min_idx = j;
                    current_max = eival_matrix(j, j);
                }
            }
            
            // 6. Set the normal
            N.row(i) = eigenvectors.row(min_idx);
        }
    }

    // suggestion : embarassignly parallel
    template <typename T>
    pair<Eigen::Matrix<T, 1, 3>, T> compute_constraint_and_value(
        const Eigen::Matrix<T, 1, 3> &v, // given point
        const Eigen::Matrix<T, 1, 3> &n, // normal for given point
        T eps, // distance factor
        T max_dist // maximum allowed distance from given point to new constraint point
    )
    {
        Eigen::Matrix<T, 1, 3> c;
        T d;
        while (true)
        { // compute a new constraint point
            c = v + n * eps;
            T dist = (c - v).norm();
            if (dist < max_dist)
            { // if the point is within allowed distance keep it
                d = (n * eps).norm();
                break; // success
            }
            else
            { // otherwise reduce eps and try again
                eps /= 2; 
            }
        }
        return make_pair(c, d);
    }

    template <typename T>
    T distance(Kdtree::CoordPoint const& a, Kdtree::CoordPoint const& b){
        size_t n = a.size();
        T result = 0.0;
        for (size_t i=0; i<n; i++)
            result += pow(a[i]-b[i], 2);
        return sqrt(result);
    }

    // suggestion : use a kdtree to find nearest neighbors
    // suggestion : assign tasks to differnet threads
    // suggestion : seems like a great fit for SIMD (inside the for loop)
    template <typename T>
    void generate_constraints_and_values(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        const Eigen::Matrix<T, -1, 3> &N, // original normals
        Eigen::Matrix<T, -1, 3> &C, // constraint points to fill
        Eigen::Matrix<T, -1, 1> &D, // constraint values to fill
        T eps // distance control between original and constraint points
    )
    {
        C = Eigen::Matrix<T, -1, 3>::Zero(V.rows()*3, V.cols()); // 3 constraints per point
        D = Eigen::Matrix<T, -1, 1>::Zero(V.rows()*3, 1); // 1 constraint value per constraint point

        // KDTree construction
        Kdtree::KdNodeVector nodes;
        for (int i = 0; i < V.rows(); i++){
            Kdtree::CoordPoint point(3);
            point = {V(i, 0), V(i, 1), V(i, 2)};
            nodes.push_back(Kdtree::KdNode(point));
        }
        Kdtree::KdTree tree(&nodes);

        for (int i = 0; i < V.rows(); i++)
        {
            Eigen::Matrix<T, 1, 3> p = V.row(i);
            Eigen::Matrix<T, 1, 3> n1 = N.row(i);
            Eigen::Matrix<T, 1, 3> n2 = -N.row(i);

            //kNN search
            Kdtree::KdNodeVector nn;
            Kdtree::CoordPoint query_point(3);
            query_point = {V(i, 0), V(i, 1), V(i, 2)};
            tree.k_nearest_neighbors(query_point, 2, &nn);
            // the second nearest is desired since the nearest is the same point
            T nn_dist = distance<T>(query_point, nn[1].point);


            // first set of constraints and points are the vertices themselves
            C.row(i) = V.row(i); // use data vertices as one set of constraints
            D(i) = 0.0; // implict function evaluates to 0 at the vertex

            // second set of constraints, points outside of the mesh
            pair<Eigen::Matrix<T, 1, 3>, T> c_d1 = compute_constraint_and_value(p, n1, eps, nn_dist);
            C.row(V.rows() + i) = c_d1.first;
            D(V.rows() + i) = c_d1.second;

            // third set of constraints, points inside the mesh
            pair<Eigen::Matrix<T, 1, 3>, T> c_d2 = compute_constraint_and_value(p, n2, eps, nn_dist);
            C.row(V.rows()*2 + i) = c_d2.first;
            D(V.rows()*2 + i) = -c_d2.second;
        }
        
    }
 
    // suggestion : ask for clarification about this function
    template <typename T>
    void generate_grid(
        Eigen::Matrix<T, -1, 3> &V, // vertices of grid to fill
        const Eigen::Matrix<int, 1, 3> &nt, // number of grid points in x, y, z directions
        const Eigen::Matrix<T, 1, 3> &gmin, // grid minimum point
        const Eigen::Matrix<T, 1, 3> &gmax, // grid maximum point
        const Eigen::Matrix<T, 1, 3> &pad // additional padding to apply to grid
    )
    {
        int nx = nt(0), ny = nt(1), nz = nt(2);
        V = Eigen::Matrix<T, -1, 3>::Zero(nx*ny*nz, 3);

        // Compute points on each coordinate
        Eigen::ArrayXf x_points = Eigen::ArrayXf::LinSpaced(nx, gmin(0) - pad(0), gmax(0) + pad(0));
        Eigen::ArrayXf y_points = Eigen::ArrayXf::LinSpaced(nx, gmin(1) - pad(1), gmax(1) + pad(1));
        Eigen::ArrayXf z_points = Eigen::ArrayXf::LinSpaced(nx, gmin(2) - pad(2), gmax(2) + pad(2));
        // Generate vertices of tet grid
        int vi = 0;
        int x_len = x_points.rows(), y_len = y_points.rows(), z_len = z_points.rows();
        for (int k = 0; k < z_len; k++)
        {
            auto z_axis = z_points(k);
            for (int j = 0; j < y_len; j++)
            {
                auto y_axis = y_points(j);
                for (int i = 0; i < x_len; i++)
                {
                    // better for vectorization
                    vi = i + j*x_len + k*y_len*x_len;
                    V.row(vi) = Eigen::RowVector3d(x_points(i), y_axis, z_axis);
                    // bad for vectorization
                    // vi++;
                }
            }
        }
    }
    
    // suggestion : seems like race condition may happen if parallelized!
    template <typename T>
    void generate_tets_from_grid(
        Eigen::Matrix<int, -1, 4> &TF, // tet definitions
        const Eigen::Matrix<T, -1, 3> &TV, // grid vertices
        const Eigen::Matrix<int, 1, 3> &nt // number of grid points in x, y, z directions
    )
    {
        int nx = nt(0), ny = nt(1), nz = nt(2);
        // each 'cube' will have 6 tets in it, and each tet is defined by 4 vertices
        TF = Eigen::Matrix<int, -1, 4>::Zero((nx-1) * (ny-1) * (nz-1) * 6, 4);

        // keep track of which 'set' of tets we are on
        int ti = 0;

        // suggestion : good for SIMD
        // we will iterate over all 'cubes' in the grid and create 6 new tets
        for (int i = 0; i < nx - 1; i++)
        {
            for (int j = 0; j < ny - 1; j++)
            {
                for (int k = 0; k < nz - 1; k++)
                {
                    // collect the indices of the vertices of this cube
                    int v1 = (i+0)+nx*((j+0)+ny*(k+0));
                    int v2 = (i+0)+nx*((j+1)+ny*(k+0));
                    int v3 = (i+1)+nx*((j+0)+ny*(k+0));
                    int v4 = (i+1)+nx*((j+1)+ny*(k+0));
                    int v5 = (i+0)+nx*((j+0)+ny*(k+1));
                    int v6 = (i+0)+nx*((j+1)+ny*(k+1));
                    int v7 = (i+1)+nx*((j+0)+ny*(k+1));
                    int v8 = (i+1)+nx*((j+1)+ny*(k+1));

                    // form 6 tets using these vertices
                    // better for SIMD - new
                    ti = i*(ny-1)*(nz-1)*6 + j*(nz-1)*6 + k*6;
                    TF.row(ti) << v1,v3,v8,v7;
                    TF.row(ti + 1) << v1,v8,v5,v7;
                    TF.row(ti + 2) << v1,v3,v4,v8;
                    TF.row(ti + 3) << v1,v4,v2,v8;
                    TF.row(ti + 4) << v1,v6,v5,v8;
                    TF.row(ti + 5) << v1,v2,v6,v8;
                    // bad for SIMD - old
                    // ti += 6; // increment our counter

                }
            }
        }
    }
    
    template <typename T>
    void polynomial_basis_vector_0(
        const Eigen::Matrix<T, 1, 3> &p, // point to compute basis vector for
        Eigen::Matrix<T, 1, -1> &b // basis vector to create
    )
    {
        T x = p(0), y = p(1), z = p(2);
        Eigen::RowVectorXd basis;
        b = Eigen::Matrix<T, 1, -1>(1);
        b << 1;
    }

    template <typename T>
    void polynomial_basis_vector_1(
        const Eigen::Matrix<T, 1, 3> &p, // point to compute basis vector for
        Eigen::Matrix<T, 1, -1> &b // basis vector to create
    )
    {
        T x = p(0), y = p(1), z = p(2);
        Eigen::RowVectorXd basis;

        b = Eigen::Matrix<T, 1, -1>(4);
        b << 1, x, y, z;

    }

     template <typename T>
    void polynomial_basis_vector_2(
        const Eigen::Matrix<T, 1, 3> &p, // point to compute basis vector for
        Eigen::Matrix<T, 1, -1> &b // basis vector to create
    )
    {
        T x = p(0), y = p(1), z = p(2);
        Eigen::RowVectorXd basis;

        b = Eigen::Matrix<T, 1, -1>(10);
        b << 1, x, y, z, x*y, x*z, y*z, x*x, y*y, z*z;

    }

    template <typename T>
    void generate_weights_matrix(
        const Eigen::Matrix<T, -1, 3> &P, // points for computation
        const Eigen::Matrix<T, 1, 3> &X, // ???what is this again????
        const float h, // height of welland function
        Eigen::Matrix<T, -1, -1> &W // weights matrix to build
    )
    {
        // Welland weights matrix is a diagonal square matrix matching the size of input points
        W = Eigen::Matrix<T, -1, -1>::Zero(P.rows(), P.rows());
        for (int i = 0; i < P.rows(); i++)
        {
            float r = (P.row(i) - X).norm();
            float w = (4 * r/h + 1)*(1 - r/h);
            w = pow(w, 4);
            W(i, i) = w;
        }
    }

    // suggestion : for-loop writes to independent elements - easy to parallelize
    template <typename T>
    void generate_basis_matrix(
        const Eigen::Matrix<T, -1, 3> &P, // points to use for generation
        Eigen::Matrix<T, -1, -1> &B // basis matrix to generate
    )
    {
        // compute basis vector for first point to determine basis matrix size
        Eigen::Matrix<T, 1, -1> bv;
        Eigen::Matrix<T, 1, 3> p {P.row(0)}; // suggestion : isn't it redundant
        polynomial_basis_vector_2<T>(p, bv); // suggestion : isn't it redundant

        B = Eigen::Matrix<T, -1, -1>::Zero(P.rows(), bv.size());
        for (int i = 0; i < P.rows(); i++)
        {
            p = P.row(i);
            polynomial_basis_vector_2(p, bv);
            B.row(i) = bv;
        }
    }

    // suggestion : once the kdtree is built, the rest can be done in parallel
    template <typename T>
    void compute_implicit_function_values(
        Eigen::Matrix<T, -1, 1> &fx, // implicit function values to compute
        const Eigen::Matrix<T, -1, 3> &TV, // vertices of grid
        const Eigen::Matrix<T, -1, 3> &C, // constraint points
        const Eigen::Matrix<T, -1, 1> &D, // values at constraint points
        float w // welland radius
    )
    {
        fx = Eigen::Matrix<T, -1, 1>::Zero(TV.rows()); // one function value per grid point
        Eigen::Matrix<T, 1, 3> p {TV.row(0)};
        Eigen::Matrix<T, 1, -1> b;
        polynomial_basis_vector_2<T>(p, b);
        int min_num_pts = b.size();

        // suggestion : construct a kd-tree and perform range-search - done!
        bool unique = false; // it doesn't matter if a node is already visited
        // constructing KDTree
        Kdtree::KdNodeVector nodes;
        for (int i = 0; i < C.rows(); i++)
        {
            Kdtree::CoordPoint point(3);
            point = {C(i, 0), C(i, 1), C(i, 2)};
            Kdtree::KdNode tmp(point);
            tmp.idx = i;
            nodes.push_back(tmp);
        }
        Kdtree::KdTree tree(&nodes);

        // evaluate the implict function at each point in the tet grid
        for (int i = 0; i < TV.rows(); i++)
        {
            Kdtree::CoordPoint point = {TV(i, 0), TV(i, 1), TV(i, 2)};
            Kdtree::KdNodeVector result;
            tree.range_nearest_neighbors(point, w, &result, unique);

            p = TV.row(i);
            // vector<int> pi = points_within_radius<T>(C, p, w);            

            // assume this point is outside of the mesh
            if(result.size() == 0)
            {
                fx(i) = 100.0;
            }
            else
            {
                vector<int> pi2(result.size());
                std::transform(result.begin(), result.end(), pi2.begin(), [](auto const& node){return node.idx;});
                // assert (pi.size() == pi2.size());
                // std::sort(pi.begin(), pi.end());
                // std::sort(pi2.begin(), pi2.end());
                // for (int idx = 0; idx < pi.size(); idx++)
                //     assert(pi[idx] == pi2[idx]);
                Eigen::Matrix<T, -1, 3> P;
                Eigen::Matrix<T, -1, 1> values;
                extract_rows<T, 3>(C, pi2, P);
                extract_rows<T, 1>(D, pi2, values);

                Eigen::Matrix<T, -1, -1> W;
                Eigen::Matrix<T, -1, -1> B;
                generate_weights_matrix<T>(P, p, w, W);
                generate_basis_matrix<T>(P, B);

                // Solve: (B.T*W*B)a = (B.T*W*D) 
                Eigen::Matrix<T, -1, -1> imd = B.transpose() * W;
                Eigen::Matrix<T, -1, -1> L;
                L = imd * B;
                Eigen::Matrix<T, -1, -1> R = imd * values;
                Eigen::Matrix<T, -1, 1> a = L.llt().solve(R);
                a = L.ldlt().solve(R);

                // Calculate function value
                Eigen::Matrix<T, 1, -1> gi;
                polynomial_basis_vector_2<T>(p, gi);
                T v = gi.dot(a);
                fx(i) = v;
            } 
        }
    }
    
    // suggestion : light computation required - embarassingly parallel
    template <typename T>
    Eigen::Matrix<T, 1, 3> generate_triangle_point(
        const Eigen::Matrix<T, 1, 3> &p1,  // point 1 of edge
        const Eigen::Matrix<T, 1, 3> &p2, // point 2 of edge
        const T v1, // value at point 1
        const T v2, // value at point 2
        T snap = 0.01 // threshold to snap vertices when close to grid edge point
    )
    {
        if (abs(v1) < snap)
        {
            return p1;
        }
        else if (abs(v2) < snap)
        {
            return p2;
        }
        else if (abs(v2-v1) < snap)
        {
            return p1;
        }
        else
        {
            float mu = v2 / (v2 - v1);
            return (mu * p1) + ((1-mu) * p2);
        }
    }
    
    // suggestion : v_i and f_i are dynamically allocated. if they are initially of size 9*|Tets| 
    // then the process can be done in parallel. However, some unused memory will be consumed.
    template <typename T>
    void marching_tetrahedra(
        const Eigen::Matrix<T, -1, 3> &G, // Tet grid vertices
        const Eigen::Matrix<int, -1, 4> &Tets, // Tets of the tet grid
        const Eigen::Matrix<T, -1, 1> &fx, // implict function values at each grid vertex
        Eigen::Matrix<T, -1, 3> &SV, // reconstructed mesh vertices
        Eigen::Matrix<int, -1, 3> &SF // reconstructed mesh faces
    )
    {

        vector<T> v_i; // holder for the mesh vertices we will build
        vector<int> f_i; // holder for the face definition
        int t = 0; // tri counter
        for (int i = 0; i < Tets.rows(); i++)
        {
            // march over each tet in our set of tets
            int config = 0; // the 'configuration' we have for the current tet

            // current tet
            Eigen::Matrix<int, 1, 4> tet = Tets.row(i);
            
            // vertices of current tet
            Eigen::Matrix<T, 1, 3> p0 = G.row(tet(0));
            Eigen::Matrix<T, 1, 3> p1 = G.row(tet(1));
            Eigen::Matrix<T, 1, 3> p2 = G.row(tet(2));
            Eigen::Matrix<T, 1, 3> p3 = G.row(tet(3));

            // implicit function values at current tet
            T v0 = fx(tet(0));
            T v1 = fx(tet(1));
            T v2 = fx(tet(2));
            T v3 = fx(tet(3));

            // suggestion - make it branchless as follows
            // Determine which type of configuration we have
            if (v0 < 0.0) {config += 1;}
            if (v1 < 0.0) {config += 2;}
            if (v2 < 0.0) {config += 4;}
            if (v3 < 0.0) {config += 8;}

            // config += 1*(v0 < 0.0) + 2*(v1 < 0.0) + 4*(v2 < 0.0) + 8*(v3 < 0.0);
            

            // Create tris based on the configuration we have
            switch (config)
            {
                case 0: case 15: // 0000 or 1111
                {
                    break;
                }
                case 1: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 2:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p1, p0, v1, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 3: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p2, v1, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = t1a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point<T>(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 4: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p2, p0, v2, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p2, p1, v2, v1);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 5: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point<T>(p2, p3, v2, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 6: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = t2a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 7:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p3, p0, v3, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p3, p2, v3, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 8:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p3, p0, v3, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p3, p2, v3, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 9:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = t2a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 10:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point<T>(p2, p3, v2, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 11:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p2, p0, v2, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p2, p1, v2, v1);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 12:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p2, v1, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point<T>(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t1a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 13:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p1, p0, v1, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 14:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point<T>(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point<T>(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point<T>(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
    
        // Note: Using conservative matrix resizing is super slow, so we
        // simply create and fill a matrix using interm vectors

        // maybe convert the above and following sections to be 2d vectors and use conversion
        // function instead for cleaner code
        SV = Eigen::Matrix<T, -1, 3>::Zero(v_i.size()/3, 3);
        SF = Eigen::Matrix<int, -1, 3>::Zero(f_i.size()/3,3);
        for(int i = 0; i < v_i.size(); i = i + 3)
        {
            SV.row(i/3) << v_i[i], v_i[i+1], v_i[i+2];
        }
        for (int i = 0; i < f_i.size(); i = i + 3)
        {
            SF.row(i/3) << f_i[i], f_i[i+1], f_i[i+2];
        }
    }

    // suggestion : not paralleizable!!! range_search requires unique results and it means that
    // iterations are dependent. Moreover, replace_values may cause race condition if merge_vertices 
    // is parallelized
    template <typename T>
    void merge_vertices(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        Eigen::Matrix<int, -1, 3> &F, // faces
        T eps, // merge threshold
        Eigen::Matrix<T, -1, 3> &V2 // new vertices
    )
    {

        // constructing KDTree
        Kdtree::KdNodeVector nodes;
        for (int i = 0; i < V.rows(); i++)
        {
            Kdtree::CoordPoint point(3);
            point = {V(i, 0), V(i, 1), V(i, 2)};
            Kdtree::KdNode tmp(point);
            tmp.idx = i;
            nodes.push_back(tmp);
        }

        Kdtree::KdTree tree(&nodes);

        vector<Eigen::Matrix<T, 1, 3>> new_verts; // Eigen conservative resize is too heavy, this should be faster
        int current_new_v_i = 0;

        bool unique =  true; // already visited points are not of our interest

        for (int i=0; i<V.rows(); i++){
            vector<int> to_merge;
            // range search on KDTree
            Kdtree::CoordPoint point = {V(i, 0), V(i, 1), V(i, 2)};
            Kdtree::KdNodeVector result;
            tree.range_nearest_neighbors(point, eps, &result, unique);
            if (result.size() > 0){ // if point was not visited
                Eigen::Matrix<T, 1, 3> P {0.0, 0.0, 0.0}; // new point
                for (int j = 0; j < result.size(); j++)
                {
                    to_merge.push_back(result.at(j).idx);
                    Eigen::Matrix<T, 1, 3> n_P;
                    n_P << result.at(j).point.at(0), result.at(j).point.at(1), result.at(j).point.at(2);
                    P += n_P;
                }
                P /= to_merge.size();
                new_verts.emplace_back(P);
                // suggestion : can't we accumulate them and call the replace_values once?
                replace_values(F, to_merge, current_new_v_i);
                current_new_v_i++;
            }
        }

        // fill V2 with new verts
        V2 = Eigen::Matrix<T, -1, 3>::Zero(new_verts.size(), V.cols());
        for (int i = 0; i < new_verts.size(); i++)
        {
            V2.row(i) = new_verts[i];
        }
    }

    template <typename T>
    void depth_first_search(
        const unordered_map<T, vector<T> > &adj, // adjacency list to search
        vector<bool> &visisted, // set of visited nodes
        vector<T> &output, // build output path through the graph
        T u // element to search for
    )
	{
        // TODO: Move this over into a graph class
		visisted[u] = true;
		output.push_back(u);

		for (T i = 0; i < adj.at(u).size(); i++)
		{
			T uu = adj.at(u)[i]; // suggestion : make a local copy of adj.at(u) instead of accessing it multiple times
			if (!visisted[uu])
			{
				depth_first_search(adj, visisted, output, uu);
			}
		}
	}

    // suggestion : make sure that vec is sorted, then perform binary search
    // TODO: move the graph related functions into a graph class
    template <typename T>
    void add_edge(
        vector<T> &vec, // connect from all of these nodes
        T dst // to this node
    )
	{
        
		if (find(vec.begin(), vec.end(), dst) == vec.end())
		{
			vec.push_back(dst);
		}
	}

    // suggestion : replace unordered_map with vector<vector<T>> so that dfs will execute faster as well!
    template <typename T> // this is probably useless, should enforce int type, unless we want to allow for size_t
	unordered_map<T, vector<T>> adjacency_list( // should be adjacency_list_from_faces or something
        const Eigen::Matrix<T, -1, 3> &M // input faces
    )
	{
        
		unordered_map<T, vector<T> > adj;
		for (int i = 0; i < M.rows(); i++)
		{
			int v1 = M(i, 0);
			int v2 = M(i, 1);
			int v3 = M(i, 2);

            // suggestion : at least, use the pointer returned by find to prevent using hash_map twice
			if (adj.find(v1) == adj.end()) { adj.insert({ v1, vector<T>() }); }
			if (adj.find(v2) == adj.end()) { adj.insert({ v2, vector<T>() }); }
			if (adj.find(v3) == adj.end()) { adj.insert({ v3, vector<T>() }); }

			add_edge(adj[v1], v2);
			add_edge(adj[v1], v3);
			add_edge(adj[v2], v1);
			add_edge(adj[v2], v3);
			add_edge(adj[v3], v1);
			add_edge(adj[v3], v2);
		}
		return adj;
	}

    // suggestion : not easy to parallelize! seen vector makes iterations dependent on each other.
    template <typename T>
	unordered_map<int, vector<int>> connected_components(
        const Eigen::Matrix<T, -1, 3> &V, // vertices
        const Eigen::Matrix<int, -1, 3> &F // face definitions used to compute edges
    ) 
	{
        // suggestion : try to get rid of unordered_maps
		unordered_map<int, vector<int> > adj = adjacency_list<int>(F); // construct adjacency list of connected edges
        unordered_map<int, vector<int>> cc;
		vector<bool> seen(V.rows(), false); // hold which items we have seen so far

		// perform depth first search on each key in the adj_list
        int idx = 0;
		for (auto kv : adj)
		{
			vector<int> output;
			depth_first_search(adj, seen, output, kv.first);
			if (output.size() > 1) // only interested on connected components with more than one vertex
			{
                cc.insert({idx, output});
                idx++;
			}
		}
        return cc;
	}

    // suggestion : new_F seems to be redundant. Why not appending to F2?
    // suggestion : if some fixed memory is allocated to F2, it can be parallelized! 
    template <typename T>
    void largest_connected_component(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        const Eigen::Matrix<int, -1, 3> &F, // original faces 
        Eigen::Matrix<int, -1, 3> &F2 // faces defining largest connected component
    )
    {
        // suggestion : try to replace unordered_map with vector
        unordered_map<int, vector<int>> cc = connected_components<T>(V, F);
        int largest_cc_size = 0;
        int largest_cc_key;
        for (auto kv : cc)
        {
            if(kv.second.size() > largest_cc_size)
            {
                largest_cc_key = kv.first;
                largest_cc_size = kv.second.size();
            }
        }
        
        vector<bool> keep(F.rows(), false); // store the faces we want to keep for quick lookup
        vector<Eigen::Matrix<int, 1, 3>> new_F; // temporary holder for the faces we want to keep

        // flag the faces we want to keep
        for (int f : cc[largest_cc_key])
        {
            keep[f] = true;
        }

        for (int i = 0; i < F.rows(); i++)
        { // note: to keep a face, all three vertices must show up in our keep map
            if(keep[F(i,0)] && keep[F(i,1)] && keep[F(i,2)])
            {
                new_F.emplace_back(F.row(i));
            }
        }

        F2 = Eigen::Matrix<int, -1, 3>::Zero(new_F.size(), 3);
        for (int i = 0; i < new_F.size(); i++)
        {
            F2.row(i) = new_F[i];
        }
    }

    template <typename T>
    pair<vector<vector<T>>, vector<vector<int>>> reconstruction(
        const vector<vector<T>> &vertices, // input data vertices
        const vector<vector<T>> &normals, // input data normals
        const int nx, // reconstruction grid resolution x-direction
        const int ny, // reconstruction grid resolution y-direction
        const int nz, // reconstruction grid resolution z-direction
        const float mesh_scale, // scale original mesh
        const float welland_radius, // welland function height parameter
        const float epsilon // distance control parameter
    ) 
    {
        Eigen::Matrix<T, -1, 3> V; // input vertices
        Eigen::Matrix<T, -1, 3> N; // input normals

        vector2d_to_matrix<T, 3>(vertices, V);
        vector2d_to_matrix<T, 3>(normals, N);

        auto t1 = high_resolution_clock::now();
        auto t2 = high_resolution_clock::now();
        duration<float, std::milli> ms_float;

        // PCA normals don't well until we find a way to correct inversions
        /*
        cout << "PCA normals... ";
        t1 = high_resolution_clock::now();
        pca_normals(V, N, 10);
        t2 = high_resolution_clock::now();
        ms_double = t2 - t1;
        cout << ms_double.count() << "ms" << endl;
        */
        
        V = V * mesh_scale; // welland weights computation performs better with scaling
        V =  V.rowwise() - V.colwise().mean(); // re-center data on origin

        // ### Step 1: Compute constraint points and values ###
        Eigen::Matrix<T, -1, 3> C; // implict function constraint points
        Eigen::Matrix<T, -1, 1> D; // implict function values at corresponding constraints
        float eps = epsilon * (V.colwise().minCoeff() - V.colwise().maxCoeff()).norm();
        t1 = high_resolution_clock::now();
        cout << "Constraints and values... ";
        generate_constraints_and_values<T>(V, N, C, D, eps);
        t2 = high_resolution_clock::now();
        ms_float = t2 - t1;
        cout << ms_float.count() << "ms" << endl;

        // ### Step 2: Generate Tet grid ###
        Eigen::Matrix<T, -1, 3> TV; //vertices of tet grid
        Eigen::Matrix<int, -1, 4> TF; // tets of tet grid
        Eigen::Matrix<int, 1, 3> num_tets {nx, ny, nz}; 
        Eigen::Matrix<T, 1, 3> gmin = V.colwise().minCoeff(); // grid minimum point
        Eigen::Matrix<T, 1, 3> gmax = V.colwise().maxCoeff(); // grid maximum point
        Eigen::Matrix<T, 1, 3> padding {eps, eps, eps}; // additional padding for the grid
        t1 = high_resolution_clock::now();
        cout << "Tet grid... ";
        generate_grid<T>(TV, num_tets, gmin, gmax, padding); // generate grid
        generate_tets_from_grid<T>(TF, TV, num_tets);
        t2 = high_resolution_clock::now();
        ms_float = t2 - t1;
        cout << ms_float.count() << "ms" << endl;

        // ### Step 3: Compute implict function values at all tet grid points ###
        Eigen::Matrix<T, -1, 1> fx;
        t1 = high_resolution_clock::now();
        cout << "Implict function values... ";
        compute_implicit_function_values<T>(fx, TV, C, D, welland_radius);
        t2 = high_resolution_clock::now();
        ms_float = t2 - t1;
        cout << ms_float.count() << "ms" << endl;

        // ### Step 4: March tets and extract iso surface ###
        Eigen::Matrix<T, -1, 3> SV; // vertices of reconstructed mesh
        Eigen::Matrix<int, -1, 3> SF; // faces of reconstructed mesh
        t1 = high_resolution_clock::now();
        cout << "Marching tetrahedra... ";
        marching_tetrahedra<T>(TV, TF, fx, SV, SF);
        t2 = high_resolution_clock::now();
        ms_float = t2 - t1;
        cout << ms_float.count() << "ms" << endl;

        // ### Cleanup ###
        Eigen::Matrix<T, -1, 3> SV2;
        t1 = high_resolution_clock::now();
        cout << "Vertex merge... ";
        merge_vertices<T>(SV, SF, 0.00001, SV2);
        t2 = high_resolution_clock::now();
        ms_float = t2 - t1;
        cout << ms_float.count() << "ms" << endl;
        Eigen::Matrix<int, -1, 3> SF2;
        t1 = high_resolution_clock::now();
        cout << "Largest connected component search... ";
        largest_connected_component<T>(SV2, SF, SF2);
        t2 = high_resolution_clock::now();
        ms_float = t2 - t1;
        cout << ms_float.count() << "ms" << endl;

        // convert data back to vector format
        vector<vector<T>> V2; // reconstructed vertices
        vector<vector<int>> F2; // reconstructed faces
        matrix_to_2dvector<T, 3>(SV2, V2);
        matrix_to_2dvector<int, 3>(SF2, F2);

        return pair<vector<vector<T>>, vector<vector<int>>>(V2, F2);
    }


}