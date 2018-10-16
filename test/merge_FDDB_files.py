import os





def get_file_names(data_dir,f_type):
    file_names = []
    nfold = 10


    for n in range(nfold):
        if f_type == 0:
            file_name = 'FDDB-fold-%02d.txt' % (n + 1)
        if f_type == 1:
            file_name = 'FDDB-fold-%02d-ellipseList.txt' % (n + 1)
        if f_type == 2:
            file_name = 'FDDB-det-fold-%02d.txt' % (n + 1)


        file_name = os.path.join(data_dir, file_name)
        file_names.append(file_name)
    #print(file_names)
    return file_names

def merge_files(file_names,output_file_name):
    output_file_name = output_file_name
    output_file = open(output_f_name,'w')
    for file_name in file_names:
        fid = open(file_name,mode = 'r')
        for line in fid:
            output_file.writelines(line)
        print(file_name,' done')
    output_file.close()




if __name__ == '__main__':
    merge_file_dir = ['../../DATA/FDDB-folds/FDDB-folds/','../../DATA/FDDB-folds/FDDB-folds/','../../DATA/FDDB_OUTPUT/']

    o_file_names = ['Fold_all.txt','FDDB-fold-all-ellipseList.txt','FDDB-det-fold-all.txt']

    for i in range(3):

        output_f_name = os.path.join(merge_file_dir[i],o_file_names [i])
    #output_f_name = os.path.join(output_dir, 'Fold-all.txt')
        file_names = get_file_names(merge_file_dir[i],i)
        merge_files(file_names,output_f_name)