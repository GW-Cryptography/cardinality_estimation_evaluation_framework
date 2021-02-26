# Copyright 2020 The Private Cardinality Estimation Framework Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates example sets and estimates cardinality multiple ways, summarizes."""
from absl import app
from absl import flags
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators import stratified_sketch
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import CascadingLegions
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import Estimator
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import SketchEstimatorConfig
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import Simulator

FLAGS = flags.FLAGS

flags.DEFINE_integer('universe_size', 1000000,
                     'The number of unique possible user-ids')
flags.DEFINE_integer(
    'number_of_sets', 15,
    'The number of sets to depulicate across, AKA the number of publishers')
flags.DEFINE_integer('number_of_trials', 1,
                     'The number of times to run the experiment')
flags.DEFINE_integer('set_size', 10000, 'The size of all generated sets')
flags.DEFINE_integer('legion_number', 12, 'The number of legions')
flags.DEFINE_integer('legion_length', 8192, 'The length of each legion')
flags.DEFINE_integer('max_frequency', 1, 'Maximum frequency to be analyzed.')


def main(argv):
  LEGION_BY_LEGION = True
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  estimator_config_cascading_legions = SketchEstimatorConfig(
      name='cascading-legions',
      sketch_factory=CascadingLegions.get_sketch_factory(
          FLAGS.legion_number, FLAGS.legion_length),
      estimator=Estimator())

  estimator_config_list = [
      estimator_config_cascading_legions,
  ]

  name_to_estimator_config = {
      'cascading_legions': estimator_config_cascading_legions,
  }

  set_generator_factory = (
      set_generator.IndependentSetGenerator
      .get_generator_factory_with_num_and_size(
          universe_size=FLAGS.universe_size,
          num_sets=FLAGS.number_of_sets,
          set_size=FLAGS.set_size))

  for estimator_method_config in estimator_config_list:
    print(f'\nCalculations for {estimator_method_config.name}')
    set_rs = np.random.RandomState(1)
    sketch_rs = np.random.RandomState(1)
    simulator = Simulator(
        num_runs=FLAGS.number_of_trials,
        set_generator_factory=set_generator_factory,
        sketch_estimator_config=estimator_method_config,
        set_random_state=set_rs,
        sketch_random_state=sketch_rs)

    if not LEGION_BY_LEGION:
      _, agg_data = simulator.run_all_and_aggregate(LEGION_BY_LEGION)
      print(f'Aggregate Statistics for {estimator_method_config.name}')
      print(agg_data)
  else:
      _ = simulator.run_all_and_aggregate(LEGION_BY_LEGION)



if __name__ == '__main__':
  app.run(main)
