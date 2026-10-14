import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

CHILDREN = {'A':['B','C'],'B':['D','E'],'C':['F','G'],'D':['H'],'H':['L']}
NODES = ['A','B','C','D','E','F','G','H','L']
POS = dict(A=(4,4),B=(2,3),C=(6,3),D=(1,2),E=(3,2),F=(5,2),G=(7,2),H=(1,1),L=(1,0))

def depth_search(goal='G',limit=None):
    """Forward-only, left-to-right recursive DFS / depth-limited DFS.
    Goal test precedes cutoff. Every root entry has depth zero.
    Events/visit history are teaching instrumentation, not minimal DFS memory.
    """
    if limit is not None and (not isinstance(limit,int) or limit<0):
        raise ValueError('Limit must be a nonnegative integer or None.')
    path=[];visits=[];events=[];peak=0
    def record(kind,node,message):
        events.append(dict(kind=kind,node=node,current=path[-1] if path else None,
                           path=list(path),visited=list(visits),depth=len(path)-1,
                           message=message))
    def walk(node):
        nonlocal peak
        path.append(node);visits.append(node);peak=max(peak,len(path))
        record('VISIT',node,'Enter node; perform goal test.')
        if node==goal:
            record('GOAL',node,'Goal found; stop this pass immediately.')
            return 'FOUND',list(path)
        if limit is not None and len(path)-1==limit:
            record('CUTOFF',node,'Depth limit reached; do not expand children.')
            path.pop();record('BACKTRACK',node,'Return after depth cutoff.')
            return 'CUTOFF',None
        children=CHILDREN.get(node,[])
        if not children:record('LEAF',node,'Non-goal leaf; return failure for this branch.')
        cutoff=False
        for child in children:
            status,route=walk(child)
            if status=='FOUND':return status,route
            cutoff=cutoff or status=='CUTOFF'
        path.pop();record('BACKTRACK',node,'Return to parent after available branches.')
        return ('CUTOFF' if cutoff else 'FAILURE'),None
    status,route=walk('A')
    return dict(method='DFS' if limit is None else 'DLS',goal=goal,limit=limit,
                status=status,route=route,visits=visits,events=events,peak=peak)

def iddfs(goal='G'):
    passes=[];limit=0
    while True:
        result=depth_search(goal,limit);passes.append(result)
        if result['status'] in ['FOUND','FAILURE']:
            return dict(goal=goal,passes=passes,status=result['status'],route=result['route'],
                        total_visits=sum(len(p['visits']) for p in passes),
                        peak=max(p['peak'] for p in passes))
        limit+=1

def event_table(result):
    return pd.DataFrame([{'Event':i+1,'Action':e['kind'],'Event node':e['node'],
                          'Current node':e['current'] or '(none)',
                          'Active path':' -> '.join(e['path']) or '(empty)',
                          'Explanation':e['message']} for i,e in enumerate(result['events'])])

def pass_table(result):
    return pd.DataFrame([{'Limit':p['limit'],'Visit order':' -> '.join(p['visits']),
                          'Node visits':len(p['visits']),'Peak active-path nodes':p['peak'],
                          'Result':p['status'],'Solution path':' -> '.join(p['route'] or []) or '(none)'}
                         for p in result['passes']])

def draw_tree(goal='G',event=None,limit=None):
    fig,ax=plt.subplots(figsize=(10,5.6))
    for node,children in CHILDREN.items():
        for child in children:
            x,y=POS[node];u,v=POS[child]
            ax.annotate('',xy=(u,v+.13),xytext=(x,y-.13),arrowprops=dict(arrowstyle='->',color='#AAB6C5',lw=1.5),zorder=1)
    if limit is not None:
        boundary=4-limit-.2
        ax.axhline(boundary,color='#BC6A15',ls='--',lw=1.5)
        ax.text(7.7,boundary+.1,f'limit {limit}',color='#BC6A15',ha='right')
    for node,(x,y) in POS.items():
        fill='#EDF2F8';edge='#2864A6'
        if event:
            if node in event['visited']:fill='#DCE6F2'
            if node in event['path']:fill='#D8F0E4';edge='#16865B'
            if node==event['current']:fill='#E9DDF7';edge='#7443AE'
            if node==event['node'] and event['kind']=='CUTOFF':fill='#FFF0DD';edge='#BC6A15'
        if node==goal:edge='#B48610'
        ax.text(x,y,node,ha='center',va='center',fontsize=13,fontweight='bold',
                bbox=dict(boxstyle='circle,pad=.45',facecolor=fill,edgecolor=edge,lw=2),zorder=3)
    for depth in range(5):ax.text(-.35,4-depth,f'depth {depth}',color='#53647B',va='center')
    title=f'Goal {goal} | '+('DFS: no depth limit' if limit is None else f'DLS: limit {limit}')
    if event:title+=f" | {event['kind']} {event['node']}"
    ax.set_title(title);ax.set_xlim(-1,8);ax.set_ylim(-.5,4.5);ax.axis('off')
    plt.tight_layout();plt.show();plt.close(fig)

def view_event(result,event=1):
    if not 1<=event<=len(result['events']):raise ValueError('Event outside this pass.')
    e=result['events'][event-1]
    print('Event',event,e['kind'],'| node',e['node'],'|',e['message'])
    print('Active path:', ' -> '.join(e['path']) or '(empty)')
    print('Visit order so far:', ' -> '.join(e['visited']))
    draw_tree(result['goal'],e,result['limit'])

def report_pass(result):
    print(result['method'],'| limit:',result['limit'],'|',result['status'])
    print('Visit order:', ' -> '.join(result['visits']))
    print('Solution path:', ' -> '.join(result['route'] or []) or '(none)')
    print('Node visits:',len(result['visits']),'| peak active-path nodes:',result['peak'])
    display(event_table(result))

def show_dfs(result):
    report_pass(result)
    try:
        import ipywidgets as widgets
        widgets.interact(lambda event:view_event(result,event),
                         event=widgets.IntSlider(min=1,max=len(result['events']),value=1,description='Event'))
    except ImportError:
        print('Use view_event(result, event=2) if widgets are unavailable.')
        view_event(result)
    return result

def show_iddfs(result):
    display(pass_table(result))
    print('Total node visits across passes:',result['total_visits'],
          '| peak active-path nodes:',result['peak'])
    print('Solution path:', ' -> '.join(result['route'] or []) or '(none)')
    try:
        import ipywidgets as widgets
        selector=widgets.Dropdown(options=[(f"Limit {p['limit']}",i) for i,p in enumerate(result['passes'])],description='Pass')
        step=widgets.IntSlider(min=1,max=len(result['passes'][0]['events']),value=1,description='Event')
        out=widgets.Output()
        def redraw(change=None):
            p=result['passes'][selector.value]
            with out:
                out.clear_output(wait=True);view_event(p,min(step.value,len(p['events'])))
        def choose(change):
            step.value=1;step.max=len(result['passes'][selector.value]['events']);redraw()
        selector.observe(choose,names='value');step.observe(redraw,names='value')
        display(widgets.VBox([selector,step,out]));redraw()
    except ImportError:
        print('Use view_event(result["passes"][2], event=1), etc.')
        view_event(result['passes'][-1])
    return result
