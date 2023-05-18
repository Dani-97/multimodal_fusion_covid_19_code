import argparse
import os

# If only_print is set to True, then the commands will not be executed, only
# printed.
# If only_run_first_command is set to True, then only the first command will be
# executed. This is useful if the first command fails, as it stops the whole
# running. Interesting for debugging purposes.
def execute_train(experiments_list):
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_print', action='store_true', \
                   help='If specified, the commands will not be executed, only printed.' + \
                        'This is very useful for debugging purposes.')
    parser.add_argument('--only_run_first_command', action='store_true', \
                   help='If specified, only the first command of the list will be ' + \
                         'printed or executed. This is useful for debugging, as if ' + \
                         'the first command fails, the execution does not continue ' + \
                         'running')
    args = parser.parse_args()

    if (args.only_run_first_command):
        experiments_list = [experiments_list[0]]
    for current_experiment_aux in experiments_list:
        command_to_execute = 'python3 train.py' + ' '
        for current_param_key_aux in current_experiment_aux.keys():
            current_param_value_aux = current_experiment_aux[current_param_key_aux]
            if (current_param_value_aux=='store_true'):
                command_to_execute += '--%s'%current_param_key_aux + ' '
            else:
                command_to_execute += '--%s %s'%(current_param_key_aux, current_param_value_aux) + ' '

        print('\n##################################################################')
        print(' Executing command [%s]'%command_to_execute)
        print('##################################################################\n')
        if (not args.only_print):
            os.system(command_to_execute)
