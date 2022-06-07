#include <igl/opengl/glfw/Viewer.h>
#include "reconstruction.h"

using namespace std;
using namespace vectorization;

int main(int argc, char *argv[])
{ 
    // read data from .off file
    Eigen::Matrix<double, -1, 3> V; // vertices of the provided data
    Eigen::Matrix<int, -1, 3> F; // triangle definitions of the provided data
    Eigen::Matrix<double, -1, 3> N; // per-vertex normals of the provided data
    igl::readOFF("../data/bunny.off", V, F);
    std::cout << "Size of V: " << V.rows() << "\n";
    igl::per_vertex_normals(V, F, N);

    // collect data into containers
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<double>> normals;
    std::vector<std::vector<double>> faces;

    // convert data to STL vectors
    matrix_to_2dvector<double, 3>(V, vertices);
    matrix_to_2dvector<double, 3>(N, normals);

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>> > R;
    R = reconstruction<double>(vertices, normals, 20, 20, 20, 200.0, 4.0, 0.01);

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