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
    # Replace all taglist nodes with a list of the collected tags.
    # Augment each tag with a backlink to the original location.
    env = app.builder.env

    # for node in doctree.traverse(taglist):
    #     node.replace_self([])
    #     continue

    #content = []


    # para = nodes.paragraph()
    # filename = env.doc2path(tag_info['docname'], base=None)
    # description = (
    #     _('(The original entry is located in %s, line %d and can be found ') %
    #     (filename, tag_info['lineno']))
    # para += nodes.Text(description, description)
    #
    # # Create a reference
    # newnode = nodes.reference('', '')
    # innernode = nodes.emphasis(_('here'), _('here'))
    # newnode['refdocname'] = tag_info['docname']
    # newnode['refuri'] = app.builder.get_relative_uri(
    #     fromdocname, tag_info['docname'])
    # newnode['refuri'] += '#' + tag_info['target']['refid']
    # newnode.append(innernode)
    # para += newnode
    # para += nodes.Text('.)', '.)')
    #
    # # Insert into the taglist
    # content.append(tag_info['tag'])
    # content.append(para)

    #node.replace_self(content)

class tag (nodes.Admonition, nodes.Element):
    pass

class TagDirective(Directive):
    #This allows content in the directive, e.g. to list tags here
    has_content = True

    def run(self):
        env = self.state.document.settings.env
        targetid = "tag-%d" % env.new_serialno('tag')
        targetnode = nodes.target('', '', ids=[targetid])

        tags = self.content
        links = []
        #self.content.append("<a href =\"../tags/junk\" >")

        # for tag in tags:
        #     link = "<a href=\"../tags/" + tag +  ".html>\"" + tag + "</a>"
        #     links.append(link)

        ad = make_admonition(tag, self.name, [_('Tags')], self.options,
                             self.content, self.lineno, self.content_offset,
                             self.block_text, self.state, self.state_machine)

        return [targetnode] + ad
