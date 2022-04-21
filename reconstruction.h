#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <igl/opengl/glfw/Viewer.h>

namespace mtr {
    using namespace std;

    template <typename T, int cols> 
    void matrix_to_2dvector(
        Eigen::Matrix<T, -1, cols> &matrix, // the matrix to convert
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
        const vector<vector<T>> &vec, // input 2d vector
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

    template <typename T>
    vector<int> points_within_radius(
        const Eigen::Matrix<T, -1, 3> &P, // the set of points to search
        const Eigen::Matrix<T, 1, 3> &origin, // origin point to compute from
        T radius // radius of ball
    )
    {
        Eigen::Matrix<T, -1, 1> distances {(P.rowwise() - origin).rowwise().norm()};
        vector<int> pi;
        for (int d = 0; d < distances.rows(); d++)
        {
            if (distances(d) < radius)
            {
                pi.push_back(d);
            }
        }
        return pi;
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

    template <typename T>
    void replace_values(
        Eigen::Matrix<T, -1, 3> &M, // matrix to update
        const vector<T> &vals, // values to replace
        T new_val // new value
    )
    { // could improve this by using a hash map instead, i.e. f_premerge => f_a
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
    
    template <typename T>
    pair<vector<int>, vector<T> > get_nearest_neighbours_with_distances(
        const Eigen::Matrix<T, -1, 3> &V, // set of points to look in
        const Eigen::Matrix<T, 1, 3> &p, // the points to compute from
        int n // the number of desired neighbors
    )
    {
        vector<int> nn;
        vector<T> nn_dist;
        
        Eigen::Matrix<T, -1, 1> distances = (V.rowwise() - p).rowwise().norm();

        // TODO: This is not the best way to do it, consider sorting the list using an insertion_sort 
        //  to partially sort the list
        // Find n minimums in distance vector
        while(nn.size() < n)
        {
            int min_idx = -1; // assume first idx is minimum
            double min_value = 100000000.0; // stupid hack, can't figure out why first point screws up
            for (int i = 0; i < V.rows(); i++)
            {
                if ((find(nn.begin(), nn.end(), i) == nn.end()) && min_idx != i && distances(i) > 0.0 && distances(i) < min_value)
                {
                    min_value = distances(i);
                    min_idx = i;
                }
            }
            nn.insert(nn.end(), min_idx);
            nn_dist.insert(nn_dist.end(), min_value);
        }
        return make_pair(nn, nn_dist);
    }

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

        for (int i = 0; i < V.rows(); i++)
        {
            Eigen::Matrix<T, 1, 3> p = V.row(i);
            Eigen::Matrix<T, 1, 3> n1 = N.row(i);
            Eigen::Matrix<T, 1, 3> n2 = -N.row(i);

            pair<vector<int>, vector<T> > nn = get_nearest_neighbours_with_distances(V, p, 1);

            // first set of constraints and points are the vertices themselves
            C.row(i) = V.row(i); // use data vertices as one set of constraints
            D(i) = 0.0; // implict function evaluates to 0 at the vertex

            // second set of constraints, points outside of the mesh
            pair<Eigen::Matrix<T, 1, 3>, T> c_d1 = compute_constraint_and_value(p, n1, eps, nn.second[0]);
            C.row(V.rows() + i) = c_d1.first;
            D(V.rows() + i) = c_d1.second;

            // third set of constraints, points inside the mesh
            pair<Eigen::Matrix<T, 1, 3>, T> c_d2 = compute_constraint_and_value(p, n2, eps, nn.second[0]);
            C.row(V.rows()*2 + i) = c_d2.first;
            D(V.rows()*2 + i) = -c_d2.second;
        }
        
    }
 
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

    template <typename T>
    void generate_basis_matrix(
        const Eigen::Matrix<T, -1, 3> &P, // points to use for generation
        Eigen::Matrix<T, -1, -1> &B // basis matrix to generate
    )
    {
        // compute basis vector for first point to determine basis matrix size
        Eigen::Matrix<T, 1, -1> bv;
        Eigen::Matrix<T, 1, 3> p {P.row(0)};
        polynomial_basis_vector(2, p, bv);

        B = Eigen::Matrix<T, -1, -1>::Zero(P.rows(), bv.size());
        for (int i = 0; i < P.rows(); i++)
        {
            p = P.row(i);
            polynomial_basis_vector(2, p, bv);
            B.row(i) = bv;
        }
    }

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
        polynomial_basis_vector(2, p, b);
        int min_num_pts = b.size();

        // evaluate the implict function at each point in the tet grid
        for (int i = 0; i < TV.rows(); i++)
        {
            Eigen::Matrix<T, 1, 3> p {TV.row(i)};
            vector<int> pi = points_within_radius(C, p, w);            

            // assume this point is outside of the mesh
            if(pi.size() <= 0)
            {
                fx(i) = 100.0;
            }
            else
            {
                Eigen::Matrix<T, -1, 3> P;
                Eigen::Matrix<T, -1, 1> values;
                extract_rows(C, pi, P);
                extract_rows(D, pi, values);

                Eigen::Matrix<T, -1, -1> W;
                Eigen::Matrix<T, -1, -1> B;
                generate_weights_matrix(P, p, w, W);
                generate_basis_matrix(P, B);

                // Solve: (B.T*W*B)a = (B.T*W*D) 
                Eigen::Matrix<T, -1, -1> imd = B.transpose() * W;
                Eigen::Matrix<T, -1, -1> L;
                L = imd * B;
                Eigen::Matrix<T, -1, -1> R = imd * values;
                Eigen::Matrix<T, -1, 1> a = L.llt().solve(R);
                a = L.ldlt().solve(R);

                // Calculate function value
                Eigen::Matrix<T, 1, -1> gi;
                polynomial_basis_vector(2, p, gi);
                T v = gi.dot(a);
                fx(i) = v;
            } 
        }
        
    }
    
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

            // implict function values at current tet
            T v0 = fx(tet(0));
            T v1 = fx(tet(1));
            T v2 = fx(tet(2));
            T v3 = fx(tet(3));

            // Determine which type of configuration we have
            if (v0 < 0.0) {config += 1;}
            if (v1 < 0.0) {config += 2;}
            if (v2 < 0.0) {config += 4;}
            if (v3 < 0.0) {config += 8;}

            // Create tris based on the configuration we have
            switch (config)
            {
                case 0: case 15: // 0000 or 1111
                {
                    break;
                }
                case 1: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 2:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p1, p0, v1, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 3: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p1, p2, v1, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = t1a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 4: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p2, p0, v2, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p2, p1, v2, v1);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 5: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point(p2, p3, v2, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 6: 
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = t2a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 7:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p3, p0, v3, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p3, p2, v3, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 8:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p3, p0, v3, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p3, p2, v3, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 9:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = t2a;
                    Eigen::Matrix<T, 1, 3> t3b = generate_triangle_point(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 10:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t2a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point(p2, p3, v2, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 11:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p2, p0, v2, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p2, p1, v2, v1);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 12:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p1, p2, v1, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    Eigen::Matrix<T, 1, 3> t1b = t3a;
                    Eigen::Matrix<T, 1, 3> t2b = generate_triangle_point(p0, p3, v0, v3);
                    Eigen::Matrix<T, 1, 3> t3b = t1a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});
                    t += 3;
                    break;
                }
                case 13:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p1, p0, v1, v0);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p1, p3, v1, v3);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    t += 3;
                    break;
                }
                case 14:
                {
                    Eigen::Matrix<T, 1, 3> t1a = generate_triangle_point(p0, p1, v0, v1);
                    Eigen::Matrix<T, 1, 3> t2a = generate_triangle_point(p0, p2, v0, v2);
                    Eigen::Matrix<T, 1, 3> t3a = generate_triangle_point(p0, p3, v0, v3);
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

    template <typename T>
    void merge_vertices(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        Eigen::Matrix<int, -1, 3> &F, // faces
        T eps, // merge threshold
        Eigen::Matrix<T, -1, 3> &V2 // new vertices
    )
    {
        // kind of a hash table where index i represents vertex i, 
        // flag sets whether we have 'seen' this vertex
        vector<bool> seen = vector<bool>(V.rows(), false); 
        vector<int> merged;
        vector<Eigen::Matrix<T, 1, 3>> new_verts; // Eigen conservative resize is too heavy, this should be faster

        int current_new_v_i = 0;
        for (int i = 0; i < V.rows(); i++)
        {
            if(!seen[i]) // Note: Only process this vert if we haven't seen it yet!
            {
                // first build a temporary set of points from the vertices we SHOULD be searching...
                vector<int> indices;
                for (int j = 0; j < V.rows(); j++)
                {
                    if(!seen[j]) { indices.emplace_back(j);}
                }

                // now that we have the indices to look through, extract rows from V into temp
                Eigen::Matrix<T, -1, 3> V_tmp;
                extract_rows(V, indices, V_tmp);
 
                // calculate distances from our current point to all points in V_tmp
                Eigen::Matrix<T, -1, 1> d = (V_tmp.rowwise() - V.row(i)).rowwise().norm();
                vector<int> to_merge;
                for (int j = 0; j < d.size(); j++)
                {
                    if(d(j) < eps)
                    {
                        to_merge.emplace_back(indices[j]);
                    }
                }

                // merge vertices, add vertex to new set and update references in F to point to new vertex set index
                Eigen::Matrix<T, 1, 3> P {0.0, 0.0, 0.0}; // new point
                if (to_merge.size() > 0) // can't do division by zero
                {
                    for (int j = 0; j < to_merge.size(); j++)
                    {
                        P += V.row(to_merge[j]);
                        seen[to_merge[j]] = true;
                    }
                    P /= to_merge.size();
                    new_verts.emplace_back(P);
                    replace_values(F, to_merge, current_new_v_i);
                    current_new_v_i++;
                }
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
		visisted[u] = true;
		output.push_back(u);

		for (T i = 0; i < adj.at(u).size(); i++)
		{
			T uu = adj.at(u)[i];
			if (!visisted[uu])
			{
				depth_first_search(adj, visisted, output, uu);
			}
		}
	}

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

    template <typename T>
	unordered_map<int, vector<int>> connected_components(
        const Eigen::Matrix<T, -1, 3> &V, // vertices
        const Eigen::Matrix<int, -1, 3> &F // face definitions used to compute edges
    ) 
	{
		unordered_map<int, vector<int> > adj = adjacency_list(F); // construct adjacency list of connected edges
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

    template <typename T>
    void largest_connected_component(
        const Eigen::Matrix<T, -1, 3> &V, // original vertices
        const Eigen::Matrix<int, -1, 3> &F, // original faces 
        Eigen::Matrix<int, -1, 3> &F2 // faces defining largest connected component
    )
    {
        unordered_map<int, vector<int>> cc = connected_components(V, F);
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
        const vector<vector<T>> &normals // input data normals
    ) 
    {
        vector<vector<T>> V2; // reconstructed vertices
        vector<vector<int>> F2; // reconstructed faces

        // TODO: Enable ability to not provide normals, 
        //   and bring PCA-based normals back (or quadratic normals)

        Eigen::Matrix<T, -1, 3> V; // input vertices
        Eigen::Matrix<T, -1, 3> N; // input normals

        vector2d_to_matrix(vertices, V);
        vector2d_to_matrix(normals, N);

        V = V * 200; // welland weights computation performs better with scaling
        V =  V.rowwise() - V.colwise().mean(); // re-center data on origin

        // ### Step 1: Compute constraint points and values ###
        Eigen::Matrix<T, -1, 3> C; // implict function constraint points
        Eigen::Matrix<T, -1, 1> D; // implict function values at corresponding constraints
        double eps = 0.01 * (V.colwise().minCoeff() - V.colwise().maxCoeff()).norm();
        generate_constraints_and_values(V, N, C, D, eps);

        // ### Step 2: Generate Tet grid ###
        Eigen::Matrix<T, -1, 3> TV; //vertices of tet grid
        Eigen::Matrix<int, -1, 4> TF; // tets of tet grid
        Eigen::Matrix<int, 1, 3> num_tets {20, 20, 20};
        Eigen::Matrix<T, 1, 3> gmin = V.colwise().minCoeff(); // grid minimum point
        Eigen::Matrix<T, 1, 3> gmax = V.colwise().maxCoeff(); // grid maximum point
        Eigen::Matrix<T, 1, 3> padding {eps, eps, eps}; // additional padding for the grid
        generate_grid(TV, num_tets, gmin, gmax, padding); // generate grid
        generate_tets_from_grid(TF, TV, num_tets);

        // ### Step 3: Compute implict function values at all tet grid points ###
        Eigen::Matrix<T, -1, 1> fx;
        double welland_radius = 3.0; // TODO: try to define this programatically
        compute_implicit_function_values(fx, TV, C, D, welland_radius);

        // ### Step 4: March tets and extract iso surface ###
        Eigen::Matrix<T, -1, 3> SV; // vertices of reconstructed mesh
        Eigen::Matrix<int, -1, 3> SF; // faces of reconstructed mesh
        marching_tetrahedra(TV, TF, fx, SV, SF);

        // ### Cleanup ###
        Eigen::Matrix<T, -1, 3> SV2;
        merge_vertices(SV, SF, 0.00001, SV2);

        Eigen::Matrix<int, -1, 3> SF2;
        largest_connected_component(SV2, SF, SF2);

        // test reconstruction
        igl::opengl::glfw::Viewer viewer;
        viewer.data().clear();
        viewer.data().set_mesh(SV2, SF2);
        viewer.launch();

        // TODO: convert data back to vector format

        return pair<vector<vector<T>>, vector<vector<int>>>(V2, F2);
    }


}