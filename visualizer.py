import streamlit as st
from streamlit_agraph import agraph,Node,Edge,Config

def display_knowledge_graph(papers):
    if not papers:
        st.warning("No papers to visualize")
    
    nodes =[]
    edges = []
    #sets to keep track of added ids to avoid duplicates
    added_nodes = set()
    added_edges = set()

    #central topic node
    root_id = "ROOT"
    nodes.append(Node(id=root_id,label="Topic",size =  25,color="#6366f1"))
     
    for i,p in enumerate(papers):
        paper_id = str(i)
        title = p.get('title','Untitled')[:20] + "..."
        #paper nodes
        if paper_id not in added_nodes:
            nodes.append(Node(id=paper_id,label=title,size=15,color="#10b981"))
            added_nodes.add(paper_id)
            #link ppaer to root
            edges.append(Edge(source=root_id,target=paper_id,color="#4b5563"))

        #author nodes and edges
        authors = p.get('authors',[])
        if isinstance(authors, list):
            for auth in authors[:2]:
                auth_id = f"AUTH_{auth.replace(' ','_')}"
            if auth_id not in added_nodes:
                nodes.append(Node(id=auth_id, label=auth , size=10,color="#f59e0b"))
                added_nodes.add(auth_id)

                edge_id = f"{paper_id}-{auth_id}"
                if edge_id not in added_edges:
                    edges.append(Edge(source=paper_id,target=auth_id,color="#9ca3af"))
                    added_edges.add(edge_id)

    #graph config
    config = Config(width="100%",height=500,directed=False,nodeHighlightBehavior=True,highlightColor="#F7A7A6",collapsible=True)

    st.markdown("### 🕸️ Research Network")
    st.caption("Connections based on Papers and Top Authors")
    agraph(nodes=nodes,edges=edges,config=config)