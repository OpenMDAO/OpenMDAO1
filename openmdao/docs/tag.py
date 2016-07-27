#tag.py
from sphinx.util.compat import Directive, make_admonition
from docutils import nodes
from sphinx.locale import _

#The setup function for the Sphinx extension
def setup(app):
    #This adds a new node class to build sys, with custom functs, (same name as file)
    app.add_node(tag, html=(visit_tag_node, depart_tag_node))
    #This creates a new ".. tags:: " directive in Sphinx
    app.add_directive('tags', TagDirective)
    #These are event handlers, functions connected to events.
    app.connect('doctree-resolved', process_tag_nodes)
    app.connect('env-purge-doc', purge_tags)
    #Identifies the version of our extension
    return {'version': '0.1'}

def visit_tag_node(self, node):
    self.visit_admonition(node)

def depart_tag_node(self, node):
    self.depart_admonition(node)

def purge_tags(app, env, docname):
        return

def process_tag_nodes(app, doctree, fromdocname):
    #Backlink tag to its tag page.
    env = app.builder.env



class tag (nodes.Admonition, nodes.Element):
    pass

class TagDirective(Directive):
    #This allows content in the directive, e.g. to list tags here
    has_content = True

    def run(self):
        env = self.state.document.settings.env
        targetid = "tag-%d" % env.new_serialno('tag')
        targetnode = nodes.target('', '', ids=[targetid])

        #the tags from the directive are sitting in self.content[0]
        taggs = self.content[0].split(", ")
        links = []

        for tagg in taggs:
            #like `Python <http://www.python.org/>`_.
            link = "`" + tagg  +" <../tags/" + tagg + ".html>`_ "
            links.append(link)
        linkjoin = ", ".join(links)

        #replace the tags with links to their tag pages.
        self.content[0] = linkjoin

        ad = make_admonition(tag, self.name, [_('Tags')], self.options,
                             self.content, self.lineno, self.content_offset,
                             self.block_text, self.state, self.state_machine)

        return [targetnode] + ad
