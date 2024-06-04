# Built-in modules
import subprocess
from io import StringIO
from typing import Literal

# Third-party modules
import pandas as pd

# Custom modules
from src.misc.logger import logger

class InteProScan:

    def __init__(self, seq: str):
        self.seq = seq
        self.job_id = None

    def run(self):
        
        # Run the job
        cmd = f"curl -X POST --header 'Content-Type: application/x-www-form-urlencoded' --header 'Accept: text/plain' -d 'email=a.sanchezcano%40uva.nl&goterms=false&pathways=false&stype=p&sequence={self.seq}' 'https://www.ebi.ac.uk/Tools/services/rest/iprscan5/run'"
        result = subprocess.run(cmd, shell = True, capture_output = True).stdout.decode()

        # Check that it returns the job id
        if 'error' in result:
            raise Exception(f"Error in the request\n`{result}`")
        else:
            self.job_id = result

        # Logging
        logger.info(f"InterProScan job ID {self.job_id}")
    
    def status(self) -> Literal['QUEUED', 'RUNNING', 'FINISHED', 'ERROR', 'NOT_FOUND', 'FAILED']:

        # Check if the job has been run
        if self.job_id is None:
            raise Exception("The job hasn't been run")
        
        # Check the status of the job
        cmd = f"curl -X GET --header 'Accept: text/plain' 'https://www.ebi.ac.uk/Tools/services/rest/iprscan5/status/{self.job_id}'"
        result = subprocess.run(cmd, shell = True, capture_output = True).stdout.decode()   
        return result
    
    def result(self) -> dict[Literal['MADS-box', 'K-box'], tuple[int, int]]:

        # Check if the job has been run
        if self.job_id is None:
            raise Exception("The job hasn't been run")
        
        # Return job results
        cmd = f"curl -X GET --header 'Accept: text/plain' 'https://www.ebi.ac.uk/Tools/services/rest/iprscan5/result/{self.job_id}/tsv'"
        result = subprocess.run(cmd, shell = True, capture_output = True).stdout.decode()
        df = pd.read_csv(StringIO(result), sep = '\t')

        # Logging
        logger.info(f"InterProScan results for sequence `{self.seq}`")
        logger.info(f'\n{df}')

        # MADS-box IPR002100
        mads_df = df[df[df.columns[11]] == 'IPR002100']
        mads_start = mads_df[df.columns[6]].min()
        mads_end = mads_df[df.columns[7]].max()

        # K-box IPR002487
        kbox_df = df[df[df.columns[11]] == 'IPR002487']
        kbox_start = kbox_df[df.columns[6]].min()
        kbox_end = kbox_df[df.columns[7]].max()

        # Return domains
        domains = {
            'MADS-box': (mads_start, mads_end),
            'K-box': (kbox_start, kbox_end)
        }

        return domains

if __name__ == "__main__":
    seq = "SSMLKTLERYQKCNYGAPETNVSTREALELSSQQEYLKLKARYEALQRSQRNLLGEDLGPLSTKELESLERQLDVSLKQIRSTRTQYMLDQLTDLQRKEHMLNEANKTLKQRLLEGTQVNQLQWNPNAQDVGYGRQQAQPQGDGFFHPLECEPTLQIGYQPDPITVAAAGPSVNNYMPGWLP" 
    ipr = InteProScan(seq)
    ipr.run()
    import time
    while True:
        status = ipr.status()
        logger.info(status)
        time.sleep(5)
        if status == 'FINISHED':
            break
    print(ipr.result())