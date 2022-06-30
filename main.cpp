#include <igl/opengl/glfw/Viewer.h>
#include "reconstruction.h"

using namespace std;
using namespace mtr;

int main(int argc, char *argv[])
{ 
    // data containers
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<double>> normals;
    std::vector<std::vector<double>> faces;
    
    /* RUN THE ALGO ON THE BUNNY */
    Eigen::Matrix<double, -1, 3> V; // vertices of the provided data
    Eigen::Matrix<int, -1, 3> F; // triangle definitions of the provided data
    Eigen::Matrix<double, -1, 3> N; // per-vertex normals of the provided data
    igl::readOFF("../data/bunny.off", V, F);
    igl::per_vertex_normals(V, F, N);
    matrix_to_2dvector<double, 3>(V, vertices);
    matrix_to_2dvector<double, 3>(N, normals);

    /* RUN THE ALGO ON A TXT FILE PRODUCED FROM utilities/data-utils.py */
    // utils::load_pts_from_file(vertices, normals, "../data/output.txt");

    /* RUN THE ALGO ON TXT FILES CONTAINING VERTICES AND NORMALS */
    // utils::load_vertices_and_normals_from_txt(vertices, normals, 78056, "../data/vertices_happy0-n=78056.txt", "../data/normals_happy0-n=78056.txt");

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>> > R;
    // R = reconstruction<double>(vertices, normals, 75, 75, 75, 200.0, 4.0, 0.01, false);
    // original bunny
    R = reconstruction<double>(vertices, normals, 20, 20, 20, 200.0, 4.0, 0.01, true);

    // reconstructed mesh in Eigen matrices
    Eigen::Matrix<double, -1, 3> V2;
    Eigen::Matrix<int, -1, 3> F2;

    // convert data into Eigen matrices for plotting
    vector2d_to_matrix<double, 3>(R.first, V2);
    vector2d_to_matrix<int, 3>(R.second, F2);

    // test reconstruction
    igl::opengl::glfw::Viewer viewer;
    viewer.data().clear();
    viewer.data().set_mesh(V2, F2);
    viewer.launch();
}