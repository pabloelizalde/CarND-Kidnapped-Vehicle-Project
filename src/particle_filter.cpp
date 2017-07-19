/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	default_random_engine gen;
	double std_x, std_y, std_psi; // Standard deviations for x, y, and psi
	std_x = std[0];
	std_y = std[1];
	std_psi = std[2];

	// Create normal distributions
	normal_distribution <double> dist_x(x, std_x);
	normal_distribution <double> dist_y(y, std_y);
	normal_distribution <double> dist_psi(theta, std_psi);

	// Set number of particles
  num_particles = 10;

	for (int i = 0; i < num_particles; i++) {

		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_psi(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  double std_x, std_y, std_yaw;
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_yaw = std_pos[2];

  for (int i = 0; i < num_particles; i++) {
  //TODO: what to do when yaw_rate es 0
    particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
    particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta - (yaw_rate * delta_t)));
    particles[i].theta += yaw_rate * delta_t;

    //define normal distributions for x,y and yaw
    normal_distribution<double> dist_x(particles[i].x, std_x);
    normal_distribution<double> dist_y(particles[i].y, std_y);
    normal_distribution<double> dist_yaw(particles[i].theta, std_yaw);

    //add gaussian noise to the predicted value for this particle
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_yaw(gen);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  weights.clear();


  for (int i = 0; i < num_particles; i++) {

    Particle particle = particles[i];

    std::vector<LandmarkObs> predictions;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    predictions.clear();
    associations.clear();
    sense_x.clear();
    sense_y.clear();

    // Transform each observation to map's coordinate system
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs new_obs;

      double observation_x = observations[j].x;
      double observation_y = observations[j].y;

      double cosine = cos(particle.theta);
      double sine = sin(particle.theta);

      new_obs.x = (observation_x * cosine) - (observation_y * sine) + particle.x;
      new_obs.y = (observation_x * sine) + (observation_y * cosine) + particle.y;

      predictions.push_back(new_obs);
    }

    // Associate with landmark positions
    for (int j = 0; j < predictions.size(); j++) {

      double pos_x = predictions[j].x;
      double pos_y = predictions[j].y;

      double minDist = std::numeric_limits<float>::max();
      int mark_id = -1;
      double mark_x = 0.0;
      double mark_y = 0.0;

      for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {

        double land_x = (double)map_landmarks.landmark_list[k].x_f;
        double land_y = (double)map_landmarks.landmark_list[k].y_f;

        double mark_dist = dist(pos_x, pos_y, land_x, land_y);

        if(mark_dist < minDist) {
          minDist = mark_dist;
          mark_id = map_landmarks.landmark_list[k].id_i;
          mark_x  = land_x;
          mark_y  = land_y;
        }
      }

      sense_x.push_back(mark_x);
      sense_y.push_back(mark_y);
      associations.push_back(mark_id);
    }

    SetAssociations(particle, associations, sense_x, sense_y);

    //Compute weights using multi-variate probability
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double totalWeight = 1.0;

    for(int j = 0; j < associations.size(); j++) {

      double mark_x = sense_x[j];
      double mark_y = sense_y[j];

      double pos_x = predictions[j].x;
      double pos_y = predictions[j].y;

      double val1 = 1 / (2 * M_PI * sig_x * sig_y);
      double val2 = ((pos_x - mark_x) * (pos_x - mark_x)) / (2 * sig_x * sig_x);
      double val3 = ((pos_y - mark_y) * (pos_y - mark_y)) / (2 * sig_y * sig_y);
      double weight = (val1 * exp(-(val2 + val3)));

      if(weight > 0) {
        totalWeight *= weight;
      }
    }

    particles[i].weight = totalWeight;
    weights.push_back(totalWeight);
  }

}

void ParticleFilter::resample() {

  default_random_engine gen;

  std::vector<Particle> newParticles;
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++) {
    int idx = dist(gen);
    newParticles.push_back(particles[idx]);
  }
  particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
