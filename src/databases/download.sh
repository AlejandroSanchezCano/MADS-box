# BioGRID
mkdir -p data/Databases/BioGRID
cd data/Databases/BioGRID
wget https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.230/BIOGRID-IDENTIFIERS-4.4.230.tab.zip 
unzip BIOGRID-IDENTIFIERS-4.4.230.tab.zip
cat BIOGRID-IDENTIFIERS-4.4.230.tab.txt | grep -E "UNIPROT-ACCESSION|TREMBL|SWISS-PROT" | cut -f -2 > uniprot-IDENTIFIERS-4.4.230.tab.txt

# STRING
mkdir -p data/Databases/STRING
cd data/Databases/STRING
wget https://stringdb-downloads.org/download/protein.aliases.v12.0.txt.gz
gzip -dk protein.aliases.v12.0.txt.gz
cat protein.aliases.v12.0.txt | grep "UniProt_AC" | cut -f -2 > uniprot.aliases.v12.0.txt

# IntAct
mkdir -p data/Databases/IntAct
cd data/Databases/IntAct
wget https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip
unzip intact.zip
wget https://ftp.ebi.ac.uk/pub/databases/intact/current/various/uniprotlinks.dat