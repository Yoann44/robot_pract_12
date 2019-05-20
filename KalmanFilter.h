#pragma once

/*
 * @file KalmanFilter.h
 * Simple Kalman Filter for variable gain low-passing
 *
 * Constant velocity model. Prediction according to
 * x_{k+1} = A * x_{k}
 * with A = [1 dt; 0 1]
 *
 * Update with a direct measurement of the first state:
 * H = [1 0]
 *
 * @author Nicolas de Palezieux (Sunflower Labs) <ndepal@gmail.com>
 *
 */

#include <../Library/Eigen/Core>

//#include <mathlib/mathlib.h>

class KalmanFilter
{
public:
	/**
	 * Default constructor, state not initialized
	 */
	KalmanFilter() {};

	/**
	 * Constructor, initialize state
	 */
	KalmanFilter(Eigen::Matrix<float, 2, 1, Eigen::DontAlign> &initial, Eigen::Matrix<float, 2, 2, Eigen::DontAlign> &covInit);

	/**
	 * Default desctructor
	 */
	virtual ~KalmanFilter() {};

	/**
	 * Initialize filter state
	 * @param initial initial state
	 * @param covInit initial covariance
	 */
	void init(Eigen::Matrix<float, 2, 1, Eigen::DontAlign> &initial, Eigen::Matrix<float, 2, 2, Eigen::DontAlign> &covInit);

	/**
	 * Initialize filter state, only specifying diagonal covariance elements
	 * @param initial0  first initial state
	 * @param initial1  second initial state
	 * @param covInit00 initial variance of first state
	 * @param covinit11 initial variance of second state
	 */
	void init(float initial0, float initial1, float covInit00, float covInit11);

	/**
	 * Predict the state with an external acceleration estimate
	 * @param dt            Time delta in seconds since last state change
	 * @param acc           Acceleration estimate
	 * @param acc_unc       Variance of acceleration estimate
	 */
	void predict(float dt, float acc, float acc_unc);

	/**
	 * Update the state estimate with a measurement
	 * @param meas    state measeasurement
	 * @param measUnc measurement uncertainty
	 * @return update success (measurement not rejected)
	 */
	bool update(float meas, float measUnc);

	/**
	 * Get the current filter state
	 * @param x1 State
	 */
	void getState(Eigen::Matrix<float, 2, 1, Eigen::DontAlign> &state);

	/**
	 * Get the current filter state
	 * @param state0 First state
	 * @param state1 Second state
	 */
	void getState(float &state0, float &state1);

	/**
	 * Get state covariance
	 * @param covariance Covariance of the state
	 */
	void getCovariance(Eigen::Matrix<float, 2, 2, Eigen::DontAlign> &covariance);

	/**
	 * Get state variances (diagonal elements)
	 * @param cov00 Variance of first state
	 * @param cov11 Variance of second state
	 */
	void getCovariance(float &cov00, float &cov11);

	/**
	 * Get measurement innovation and covariance of last update call
	 * @param innov Measurement innovation
	 * @param innovCov Measurement innovation covariance
	 */
	void getInnovations(float &innov, float &innovCov);

private:
	Eigen::Matrix<float, 2, 1, Eigen::DontAlign> _x; // state

	Eigen::Matrix<float, 2, 2, Eigen::DontAlign> _covariance; // state covariance

	float _residual; // residual of last measurement update

	float _innovCov; // innovation covariance of last measurement update
};
