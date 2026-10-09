"""StructSplice: RNA secondary-structure features for exon-inclusion (PSI) prediction.

Three BiLSTM feature configurations over the Liao/Sudarshan/Regev (PNAS 2023)
splicing library:

    A  sequence only                        ->  4 channels
    B  sequence + RNAfold dot-bracket       ->  7 channels
    C  sequence + RNAplfold accessibility   -> 12 channels
"""

__version__ = "0.1.0"
