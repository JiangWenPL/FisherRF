from __future__ import print_function

import os
import signal
import sys

class ClusterStateManager:
    """
    Cluster State Manager that handler premption
    """
    def __init__(self, time_to_run=4*3600):
        self.job_id = os.environ.get("SLURM_JOB_ID", None)

        job_qos = os.environ.get("SLURM_JOB_QOS", "kd-high")
        job_name = os.environ.get("SLURM_JOB_NAME", "bash")

        # We will not be preempted when in kd-high so not need to use this script
        self.on_cluster = self.job_id is not None and job_qos != "kd-high"  and job_name != "bash"
        if self.on_cluster:
            print("[INFO]: cluster state manager is in effect")
        else:
            print("[INFO]: cluster state manager is *NOT* in effect")

        self.external_exit = None
        self.timer_exit = False

        if self.on_cluster:
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)
            # signal.signal(signal.SIGALRM, self.timer_handler)
            signal.signal(signal.SIGUSR1, self.timer_handler)
            # signal.alarm(time_to_run)

    def signal_handler(self, signal, frame):
        print("Received signal [", signal, "]")
        self.external_exit = signal

    def timer_handler(self, signal, frame):
        print("Received alarm [", signal, "]")
        self.timer_exit = True

    def should_exit(self):
        if self.timer_exit:
            return True

        if self.external_exit is not None:
            return True

        return False

    def get_exit_code(self):
        if self.timer_exit:
            return 3

        if self.external_exit is not None:
            return 0

        return 0
    
    def requeue(self):
        if self.on_cluster:
            exit_code = self.get_exit_code()
            if exit_code != 0:
                os.system(f"scontrol requeue {self.job_id}")
            sys.exit(exit_code)