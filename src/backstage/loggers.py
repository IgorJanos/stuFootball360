#------------------------------------------------------------------------------
#
#   Football-360
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import wandb


#------------------------------------------------------------------------------
#   Logger baseclass
#------------------------------------------------------------------------------

class Logger:
    def trainingStart(self):
        pass

    def trainingEnd(self):
        pass

    def epochStart(self, epoch, stats):
        pass

    def epochEnd(self, epoch, stats):
        pass

#------------------------------------------------------------------------------
#   Loggers baseclass
#------------------------------------------------------------------------------

class Loggers(Logger):
    def __init__(self, loggers=[]):
        self.loggers = loggers

    def trainingStart(self):
        for l in self.loggers:
            l.trainingStart()

    def trainingEnd(self):
        for l in self.loggers:
            l.trainingEnd()

    def epochStart(self, epoch, stats):
        for l in self.loggers:
            l.epochStart(epoch, stats)

    def epochEnd(self, epoch, stats):
        for l in self.loggers:
            l.epochEnd(epoch, stats)


#------------------------------------------------------------------------------
#   CsvLogger baseclass
#------------------------------------------------------------------------------

class CsvLogger(Logger):
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.lines = 0
        self.separator = ','

    def trainingStart(self):
        self.lines = 0
        self.file = open(self.filename, "w")

    def trainingEnd(self):
        if (self.file is not None):
            self.file.close()
            self.file = None

    def epochEnd(self, epoch, stats):
        if (self.file is not None):
            line = ""
            s = stats.getAvg()

            if (self.lines == 0):
                # CSV header
                line = self.separator.join(list(s.keys()))
                self.file.write(line + "\n")

            # Join all values
            values = [ "{}".format(s[k]) for k in s.keys() ]
            line = self.separator.join(values)
            self.file.write(line + "\n")
            self.file.flush()
            self.lines += 1


#------------------------------------------------------------------------------
#   WandBLogger baseclass
#------------------------------------------------------------------------------

class WandBLogger(Logger):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.run = None

    def trainingStart(self):
        wandb.config = self.config
        self.run = wandb.init(config=self.config, **self.kwargs)

    def trainingEnd(self):
        if (self.run is not None):
            wandb.finish()
            self.run = None

    def epochEnd(self, epoch, stats):
        s = stats.getAvg()
        self.run.log(s)

