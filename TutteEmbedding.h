#pragma once

#include <string>
#include <iostream>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>

class TutteEmbedder {

public:
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


    static void GenerateUvMapping(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& N) {

        TutteEmbedder::tutte(V, F, U);


        const auto normalizeToZeroToOne = [](Eigen::MatrixXd& In) {
            float min = In.minCoeff();
            float max = In.maxCoeff();

            In = (In.array() - min) / (max - min);
        };

        const auto normalize = [](Eigen::MatrixXd& U)
        {
            U.rowwise() -= U.colwise().mean().eval();
            U.array() /=
                (U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff() / 2.0;
        };

        igl::per_vertex_normals(V, F, N);

        normalize(V);
        normalizeToZeroToOne(U);

    }

};