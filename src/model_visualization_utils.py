import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def generate_attention_plot(c_index, data, attention_vec_out, use_sources, attention_name,
                                      sources_out=None, save=False, display_xticks=True, disp_intemidate=False,
                                      full_attention_matrix=False, path_prefix=None, rot=0):
    if attention_name not in ['SDPA', 'SASA', 'SDPA_QS_KS']:
        print(attention_name, "is not supported")
        return
    
    fig, _ = plt.subplots()
    fig.set_size_inches(20, 1)

    c_data = data.iloc[c_index,:]
    c_hopnum = c_data['num_hops']
    c_countries = c_data['geoCC']
    c_asns = c_data['ASN']
    
#     print(c_countries)
#     print(c_asns)

    max_len = len(attention_vec_out)
    
    if full_attention_matrix:
        fig.set_size_inches(14, 12)
        disp = attention_vec_out[:c_hopnum,:c_hopnum]
        ax = sns.heatmap(disp, linewidth=0.5, cmap='hot', annot=True, fmt='.2f')
        plt.ylim(0, c_hopnum)
        plt.yticks(0.5 + np.array(range(c_hopnum)), c_countries[c_hopnum-1::-1], rotation=0)
        plt.xticks(np.array(range(c_hopnum)) + 0.5, [", AS".join([country, str(c_asns[i])]) for i, country in enumerate(c_countries)], rotation=rot)
#         plt.xticks(0.5 + np.array(range(c_hopnum)), c_countries[:c_hopnum], rotation=0)
    
    else:
        if use_sources:
            c_u = np.mean(sources_out,axis=1).reshape(1, max_len)

            if attention_name == "SDPA-QS-KS":
                c_m = np.mean(attention_vec_out, axis=0).reshape(1, max_len)
                # disp = np.concatenate((c_m, c_u))
                disp = np.concatenate((c_u, c_m))
                disp = disp[:,:c_hopnum]
                # disp[0] /= np.sum(disp[0])
                disp[1] /= np.sum(disp[1])
            else: # attention_name == "SASA":
                c_a = np.mean(attention_vec_out, axis=0).reshape(1, max_len)
                c_m = c_a * c_u
                if disp_intemidate:
                    # disp = np.concatenate((c_m, c_u, c_a))
                    disp = np.concatenate((c_a, c_u, c_m))
                    disp = disp[:,:c_hopnum]
                    # disp[2] /= np.sum(disp[2])
                    # disp[0] = disp[2]*disp[1]
                    # disp[0] /= np.sum(disp[0])
                    disp[0] /= np.sum(disp[0])
                    disp[2] = disp[0]*disp[1]
                    disp[2] /= np.sum(disp[2])
                else:
                    # disp = np.concatenate((c_a, c_u))
                    disp = np.concatenate((c_u, c_a))
                    disp = disp[:,:c_hopnum]
                    # disp[0] /= np.sum(disp[0])
                    disp[1] /= np.sum(disp[1])
    #                 # In case we want to display c_m
    #                 disp[0] = disp[2]*disp[1]
    #                 disp[0] /= np.sum(disp[0])



            ax = sns.heatmap(disp, linewidth=0.5, cmap='hot', annot=True, fmt='.2f')
            if attention_name == "SDPA-QS-KS" or not disp_intemidate:
                plt.ylim(0, 2) 
                plt.yticks([0.5, 1.5], ('Sources', 'Attenetion'), rotation=0)
            else:
                plt.ylim(0, 3) 
                plt.yticks([0.5, 1.5, 2.5], ('Attenetion', 'Sources', 'A-Mul-S'), rotation=0)
        else:
            disp = np.mean(attention_vec_out, axis=0).reshape(1, max_len)
            disp = disp[:,:c_hopnum]
            disp /= np.sum(disp)
            ax = sns.heatmap(disp[:,:c_hopnum], linewidth=0.5, cmap='hot', annot=True, fmt='.2f')

            plt.ylim(-0.5, 1.5) 
            plt.yticks([0, 0.5, 1], ('', 'Attenetion', ''), rotation=0)

        if display_xticks:
            plt.xticks(np.array(range(c_hopnum)) + 0.5, [", AS".join([country, str(c_asns[i])]) for i, country in enumerate(c_countries)], rotation=rot)
        else:
            plt.xticks([])
        
    if save:
        plt.savefig(path_prefix +  "_attention_sources_visualization_route_" 
                    + str(c_index) + "_" + attention_name, bbox_inches='tight')
    plt.show()