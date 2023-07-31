#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "voxel/VoxelGrid.h"

class ImplicitSurface
{
public:
	virtual double Eval(const Eigen::Vector3d& x) = 0;
};

class FunctionSamples
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		FunctionSamples() {}

	void insertSample(const Eigen::Vector3d& pos, const double targetFunctionValue)
	{
		m_pos.push_back(pos);
		m_val.push_back(targetFunctionValue);
	}

	std::vector<Eigen::Vector3d> m_pos;
	std::vector<double> m_val;
};

class RBF : public ImplicitSurface
{
public:
	RBF(Eigen::MatrixXd V, Eigen::MatrixXd N)
	{

		// Create function samples
		double eps = 0.01f;

		// on surface points (-> center points of the RBFs)
		for (unsigned int i = 0; i < V.rows(); i++)
		{
			const Eigen::Vector3d pt = V.row(i);
			m_funcSamp.insertSample(pt, 0); // on surface point => distance = 0
		}

		// off surface points
		for (unsigned int i = 0; i < V.rows(); i++)
		{
			const Eigen::Vector3d pt = V.row(i);
			Eigen::Vector3d n = N.row(i);

			m_funcSamp.insertSample(pt + n * eps, eps);// off surface point => distance = eps
			eps *= -1;
		}

		m_numCenters = (unsigned int)V.rows();
		std::cout << "m_numCenters: " << m_numCenters << std::endl;
		const unsigned int dim = m_numCenters + 4;

		// build and solve the linear system of equations
		m_systemMatrix = Eigen::MatrixXd(dim, dim);
		m_rhs = Eigen::VectorXd(dim);
		m_coefficents = Eigen::VectorXd(dim); // result of the linear system
		BuildSystem();
		SolveSystem();
	}

	double Eval(const Eigen::Vector3d& _x)
	{
		// TODO: eval the RBF function based on the coefficents stored in m_coefficents
		// the first m_numCenters entries contain the coefficients of the kernels (that can be evaluated using EvalBasis())
		// the following parameters are the coeffients for the linear and the constant part
		// the centers of the RBFs are the first m_numCenters sample points (use m_funcSamp.m_pos[i] to access them)
		// hint: Eigen provides a norm() function to compute the l2-norm of a vector (e.g. see macro phi(i,j))
		double result = 0.0;
		for (int i = 0; i < m_numCenters; i++)
		{
			result += m_coefficents[i] * EvalBasis((m_funcSamp.m_pos[i] - _x).norm());
		}

		result += m_coefficents[m_numCenters] * _x[0] + m_coefficents[m_numCenters + 1] * _x[1] + m_coefficents[m_numCenters + 2] * _x[2] + m_coefficents[m_numCenters + 3];

		return result;
	}

private:

	double EvalBasis(double x)
	{
		return x * x * x;
	}

#define phi(i,j) EvalBasis((m_funcSamp.m_pos[i]-m_funcSamp.m_pos[j]).norm())

	//! Computes the system matrix.
	void BuildSystem()
	{
		std::cout << "Building system" << std::endl;
		Eigen::MatrixXd A(2 * m_numCenters, m_numCenters + 4);
		Eigen::VectorXd b(2 * m_numCenters);
		A.setZero();
		b.setZero();

		// TODO fill the matrix A and the vector b as described in the exercise sheet
		// note that all sample points (both on and off surface points) are stored in m_funcSamp
		// you can access matrix elements using for example A(i,j) for the i-th row and j-th column
		// similar you access the elements of the vector b, e.g. b(i) for the i-th element
		for (int i = 0; i < 2 * m_numCenters; i++)
		{
			for (int j = 0; j < m_numCenters; j++)
			{
				A(i, j) = phi(i, j);
			}

			A(i, m_numCenters) = m_funcSamp.m_pos[i][0];
			A(i, m_numCenters + 1) = m_funcSamp.m_pos[i][1];
			A(i, m_numCenters + 2) = m_funcSamp.m_pos[i][2];
			A(i, m_numCenters + 3) = 1;

			b(i) = m_funcSamp.m_val[i];
		}


		// build the system matrix and the right hand side of the normal equation
		m_systemMatrix = A.transpose() * A;
		m_rhs = A.transpose() * b;

		// regularizer -> smoother surface
		// pushes the coefficients to zero
		double lambda = 0.0001;
		m_systemMatrix.diagonal() += lambda * lambda * Eigen::VectorXd::Ones(m_numCenters + 4);
	}

	void SolveSystem()
	{
		std::cout << "Solving RBF System" << std::endl;
		std::cout << "Computing LU..." << std::endl;

		Eigen::FullPivLU<Eigen::MatrixXd> LU(m_systemMatrix);
		m_coefficents = LU.solve(m_rhs);

		std::cout << "Done." << std::endl;
	}

	//! The given function samples (at each function sample, we place a basis function).
	FunctionSamples m_funcSamp;

	//! The number of center = number of function samples.
	unsigned int m_numCenters;

	//! the right hand side of our system of linear equation. Unfortunately, float-precision is not enough, so we have to use double here.
	Eigen::VectorXd m_rhs;

	//! the system matrix. Unfortunately, float-precision is not enough, so we have to use double here
	Eigen::MatrixXd m_systemMatrix;

	//! store the result of the linear system here. Unfortunately, float-precision is not enough, so we have to use double here
	Eigen::VectorXd m_coefficents;
};