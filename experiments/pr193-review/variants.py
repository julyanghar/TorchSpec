import os
import ast
import hashlib
import subprocess
from pathlib import Path
import torchspec.models.eagle3 as model

ROOT=Path(__file__).parent
REPO=Path(os.environ['TORCHSPEC_REPO'])
REFS={'U':'fc28d35039eaa693c1b3ab0eb1dacbf2db6a4766','P':'db66ff6042e07d698c397d2f0efe6efaedc674b7'}

def source_for(arm):
    if arm=='E':
        base=source_for('U').replace('def compute_target_p_padded(', 'def _dense_reference(')
        return base+"\n@torch.no_grad()\ndef compute_target_p_padded(target_hidden_states, target_lm_head_weight, t2d, loss_mask, length, chunk_size=4096):\n    out = _dense_reference(target_hidden_states, target_lm_head_weight, t2d, loss_mask, length, chunk_size)\n    out.target_p_padded[:, :loss_mask.shape[-1]].masked_fill_(~loss_mask.bool().unsqueeze(-1), 0)\n    return out\n"
    text=subprocess.check_output(['git','show',REFS['P' if arm=='S' else arm]+':torchspec/models/eagle3.py'],cwd=REPO,text=True)
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='compute_target_p_padded')
    if arm=='S':
        class Simplify(ast.NodeTransformer):
            def visit_If(self,n):
                if isinstance(n.test,ast.UnaryOp) and isinstance(n.test.op,ast.Not) and isinstance(n.test.operand,ast.Name) and n.test.operand.id=='all_valid':
                    return n.body
                if isinstance(n.test,ast.Name) and n.test.id=='all_valid':return n.orelse
                return self.generic_visit(n)
            def visit_Assign(self,n):
                if any(isinstance(t,ast.Name) and t.id=='all_valid' for t in n.targets):return None
                return self.generic_visit(n)
        node=Simplify().visit(node)
    return ast.unparse(ast.fix_missing_locations(node))+'\n'

def function(arm):
    source=source_for(arm);ns=dict(vars(model));exec(compile(source,str(ROOT/(arm+'-function.py')),'exec'),ns)
    return ns['compute_target_p_padded']

if __name__=='__main__':
    for arm in ['U','P','S']:
        text=source_for(arm);(ROOT/(arm+'-function.py')).write_text(text)
        print(arm,hashlib.sha256(text.encode()).hexdigest())
