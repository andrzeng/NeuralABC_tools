def EigenGame2_np(data, n, num_epochs=3000, stepsize=100, debug=True):
    M = np.matmul(data.T, data)
    M = np.array(M, dtype='float32')
    dim = M.shape[0]
    rewards = M / dim
    assert(M.dtype=='float32')
    vecs = np.ones((n, dim), dtype='float32')
    
    
    step = stepsize
    print(vecs.dtype)
    for epoch in range(num_epochs): #number of iterations
        gradients = []
        
        for index in range(len(vecs)):
            
            rewards2 = np.matmul(rewards, vecs[index])
            assert(rewards2.dtype == 'float32')
            
            r_sum = np.array(np.ones(rewards2.shape[0]), dtype='float32')
            for ancestor in range(index):

                coefficient = np.dot(np.matmul(data,vecs[index]), np.matmul(data, vecs[ancestor]))

                r_sum += coefficient * vecs[ancestor]

            r_sum = r_sum / dim
            gr = rewards2 - r_sum
            rgr = gr - np.dot(gr, vecs[index]) * vecs[index]
            gradients.append(step * rgr)
                
        if(epoch % int(num_epochs/10) == 0):
            step = step / 10
            if(debug):
                print("reducing step size; it is now ", step)
       
        for i in range(len(gradients)):
            vecs[i] = vecs[i] + gradients[i]
            vecs[i] = vecs[i] / (np.linalg.norm(vecs[i]))
    
    return M, vecs
