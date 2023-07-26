#pragma onc

#include <string>
#include <iostream>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>

class TutteEmbedder {


    static void tutte(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& U)
    {
        Eigen::VectorXi bL;
        igl::boundary_loop(F, bL);

        Eigen::MatrixXd UV;
        igl::map_vertices_to_circle(V, bL, UV);

        Eigen::SparseMatrix<double> L(V.rows(), V.rows());
        igl::cotmatrix(V, F, L);

        igl::min_quad_with_fixed_data<double> data;
        igl::min_quad_with_fixed_precompute(L, bL, Eigen::SparseMatrix<double>(), false, data);

        Eigen::VectorXd B = Eigen::VectorXd::Zero(data.n, 1);
        igl::min_quad_with_fixed_solve(data, B, UV, Eigen::MatrixXd(), U);
        U.col(0) = -U.col(0);
    }


public:
    static void GenerateUvMapping(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& N) {

        Eigen::MatrixXd U_tutte;

        tutte(V, F, U_tutte);

        // Fit parameterization in unit sphere
        const auto normalize = [](Eigen::MatrixXd& U)
        {
            U.rowwise() -= U.colwise().mean().eval();
            U.array() /=
                (U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff() / 2.0;
        };
        normalize(V);
        normalize(U_tutte);

        U = U_tutte;

        igl::per_vertex_normals(V, F, N);

    }


    // static void CreateTextureArray(Eigen::MatrixXd& U, Eigen::MatrixXd Colors)

};